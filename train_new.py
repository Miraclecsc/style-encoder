import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from my_clip.modeling_clip import CLIPTextModel
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
import wandb  # 添加wandb支持
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR  # 添加学习率调度器
import numpy as np
from collections import defaultdict

# -------------------------------
# 1. 路径和超参数设置
# -------------------------------
image_data_root = '/data2/changshuochen/model/style30k/images'
prompts_root = '/data2/changshuochen/model/style30k/image_captions.json'
model_dir = '/data2/changshuochen/model/encoder-models'
vision_mapping_model_path = os.path.join(model_dir, "vision_mapping_model.pt")
stable_diffusion_path = '/data2/changshuochen/model/stable-diffusion-v1-4'
clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
device = "cuda:0"

batch_size = 4
num_epochs = 5
learning_rate = 2e-4

# 初始化wandb
wandb.init(project="style-encoder-finetuning-new", 
           config={
               "learning_rate": learning_rate,
               "batch_size": batch_size,
               "epochs": num_epochs,
               "model": "vision-mapping-model"
           })

# -------------------------------
# 2. 数据预处理和数据集定义
# -------------------------------
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class ImageCaptionDataset(Dataset):
    def __init__(self, image_data_root, prompts_root, transform=None):
        self.image_data_root = image_data_root
        self.transform = transform
        with open(prompts_root, "r") as f:
            self.data = json.load(f)
        # 存储所有图片路径和索引的映射
        self.image_paths = [sample["image_path"] for sample in self.data]
        self.path_to_idx = {path: idx for idx, path in enumerate(self.image_paths)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_data_root, sample["image_path"])
        caption = sample["caption"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = torch.randn(3, image_size, image_size)
        
        return {"image": image, "caption": caption, "image_path": sample["image_path"], "idx": idx}
    
    def get_image_by_path(self, image_path):
        """根据图片路径获取图片"""
        full_path = os.path.join(self.image_data_root, image_path)
        try:
            image = Image.open(full_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            image = torch.randn(3, image_size, image_size)
        return image
    
class PairAssigner:
    """为每个epoch分配图像对，确保每个样本可以遍历到其他所有样本"""
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.current_epoch = 0
        
    def assign_pairs(self):
        """为当前epoch分配图像对"""
        np.random.seed(self.current_epoch)  # 确保不同epoch有不同的随机配对
        indices = np.arange(self.dataset_size)
        other_indices = np.arange(self.dataset_size)
        np.random.shuffle(other_indices)
        
        # 确保没有样本被分配给自己
        for i in range(self.dataset_size):
            if indices[i] == other_indices[i]:
                j = (i + 1) % self.dataset_size
                other_indices[i], other_indices[j] = other_indices[j], other_indices[i]
        
        self.current_epoch += 1
        return {int(indices[i]): int(other_indices[i]) for i in range(self.dataset_size)}

# 创建数据集和数据加载器
dataset = ImageCaptionDataset(image_data_root, prompts_root, transform=transform)
pair_assigner = PairAssigner(len(dataset))

def create_epoch_dataloader(dataset, batch_size, pair_assignments):
    """创建包含配对图像的dataloader"""
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["caption"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        indices = [item["idx"] for item in batch]
        
        # 获取配对的其他图像
        other_images = []
        other_image_paths = []
        for idx in indices:
            other_idx = pair_assignments[idx]
            other_path = dataset.image_paths[other_idx]
            other_image_paths.append(other_path)
            other_images.append(dataset.get_image_by_path(other_path))
        
        other_images = torch.stack(other_images)
        
        return {
            "image": images, 
            "caption": captions, 
            "image_path": image_paths,
            "other_image": other_images,
            "other_image_path": other_image_paths
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                     collate_fn=collate_fn, num_workers=4)

text_encoder = CLIPTextModel.from_pretrained(clip_path).to(device)
for param in text_encoder.parameters():
    param.requires_grad = False

# VAE 模型
vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)
for param in vae.parameters():
    param.requires_grad = False

# UNet 模型
unet = UNet2DConditionModel.from_pretrained(stable_diffusion_path, subfolder="unet").to(device)
for param in unet.parameters():
    param.requires_grad = False

# 调度器
scheduler = PNDMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

# -------------------------------
# 4. 定义 vision mapping model （用于将图像映射到 8*hidden_size 的 attribute vector）
# -------------------------------
class VisionMappingModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # 使用 CLIP 模型的 vision 部分
        self.vision_model = clip_model.vision_model
        # 根据 CLIP vision model 的 hidden_size 构建投影头
        hidden_size = self.vision_model.config.hidden_size
        projection_dim = self.vision_model.config.projection_dim
        # 我们希望映射到 8 个 token，每个维度为 hidden_size，即输出 shape [B, 8, hidden_size]
        self.proj = nn.Linear(hidden_size, 8 * projection_dim)

    def forward(self, pixel_values):
        # 得到 vision model 输出
        # CLIPVisionModel 返回一个 BaseModelOutputWithPooling，其中 pooler_output shape: [B, hidden_size]
        outputs = self.vision_model(pixel_values)
        pooled = outputs.pooler_output  # shape: [B, hidden_size]
        # 经过投影头
        proj_out = self.proj(pooled)  # shape: [B, 8*hidden_size]
        # reshape为 [B, 8, hidden_size]
        mapping = proj_out.view(-1, 8, self.vision_model.config.projection_dim)
        return mapping

# 检查模型文件是否存在
if not os.path.exists(vision_mapping_model_path):
    raise FileNotFoundError(f"Vision mapping model not found at {vision_mapping_model_path}")


def calculate_style_loss(generated_latents, style_image, vision_mapping_model, vae):
    """计算风格损失"""
    # 从生成的latents解码得到图像
    with torch.no_grad():
        generated_images = vae.decode(generated_latents / vae.config.scaling_factor).sample
        
        # 添加分辨率调整
        resize_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # VAE输出的图像范围是[-1,1]，这里不需要再次归一化，只需要调整大小
        generated_images = resize_transform(generated_images)
    
    # 计算生成图像的风格向量
    generated_style = vision_mapping_model(generated_images)
    # 计算风格参考图像的风格向量
    style_reference = vision_mapping_model(style_image)
    
    # 计算风格损失（MSE）
    style_loss = F.mse_loss(generated_style, style_reference)
    return style_loss

def calculate_content_loss(generated_latents, content_image, clip_model, vae):
    """计算内容损失"""
    # 从生成的latents解码得到图像
    if not hasattr(calculate_content_loss, "global_counter"):
        calculate_content_loss.global_counter = 0
        
    with torch.no_grad():
        generated_images = vae.decode(generated_latents / vae.config.scaling_factor).sample
        #保存这个images为图片：
        print("Generated images shape (before resize):", generated_images.shape)
        generated_image = generated_images.permute(0, 2, 3, 1).cpu().numpy()
        generated_image = (generated_image + 1) / 2
        generated_image = (generated_image * 255).astype(np.uint8)
        for i, img in enumerate(generated_image):
            global_idx = calculate_content_loss.global_counter
            Image.fromarray(img).save(f'./tmp/generated_{global_idx}.png')
            calculate_content_loss.global_counter += 1
        
        # 添加分辨率调整
        resize_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        generated_images = resize_transform(generated_images)
    
    # 使用CLIP模型提取特征
    with torch.no_grad():
        # 注意：CLIP的预处理可能与我们的不同，这里假设已经适当处理
        content_features = clip_model.get_image_features(content_image)
        generated_features = clip_model.get_image_features(generated_images)
    
    # 计算内容损失（余弦相似度的负值）
    content_features = F.normalize(content_features, dim=-1)
    generated_features = F.normalize(generated_features, dim=-1)
    content_loss = 1 - torch.sum(content_features * generated_features, dim=-1).mean()
    return content_loss

def diffusion_generation_step(unet, latents, timesteps, encoder_hidden_states):
    """执行有限步数的diffusion生成过程"""
    # 这里选择一个中间步骤进行生成，可以平衡计算效率和生成质量
    t = timesteps[0].item() // 2  # 使用中间步骤
    mid_timestep = torch.ones(latents.shape[0], device=latents.device).long() * t
    
    # 去噪预测
    noise_pred = unet(latents, mid_timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
    
    # 简化的去噪步骤，得到更接近清晰图像的latents
    alpha_t = scheduler.alphas_cumprod[t]
    sqrt_alpha_t = alpha_t ** 0.5
    sqrt_one_minus_alpha_t = (1 - alpha_t) ** 0.5
    
    # 简单的去噪公式
    denoised_latents = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    return denoised_latents

from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained(clip_path)
clip_model = clip_model.to(device)
# 加载预训练的 vision mapping model
vision_mapping_model = VisionMappingModel(clip_model).to(device)
vision_mapping_model.load_state_dict(torch.load(vision_mapping_model_path, map_location=device))
# 精调时将 vision mapping model 置为训练模式
vision_mapping_model.train()

optimizer = AdamW(vision_mapping_model.parameters(), lr=learning_rate, eps=1e-8)
# 添加学习率调度器


# -------------------------------
# 5. 定义 get_final_embeddings 函数（用于将 attribute vector 注入到文本 embedding 中）
# -------------------------------
def get_final_embeddings(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    # 原始 token embeddings: [B, seq_len, hidden_size]
    original_token_embeddings = token_embed(input_ids)
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])
    if attribute_vector is not None and pos is not None:
        batch_size, num_attr, hidden_size = attribute_vector.shape
        for i in range(batch_size):
            start = pos[i].item() if torch.is_tensor(pos[i]) else pos[i]
            # 确保start位置不会导致索引越界
            if start + 10 > input_ids.shape[1]:
                start = input_ids.shape[1] - 10
            
            # 保存原本 start+1:start+3 的标点（如句号和 eos）
            end_embeddings = original_token_embeddings[i, start + 1: start + 3, :].clone()
            # 用 attribute_vector 替换从 start 开始的连续 8 个 token
            original_token_embeddings[i, start: start + 8, :] = attribute_vector[i]
            # 将原来的标点放到 start+8:start+10
            original_token_embeddings[i, start + 8: start + 10, :] = end_embeddings
    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

# -------------------------------
# 6. 精调训练循环：标准 noise loss（MSE Loss）
# -------------------------------
style_weight = 100.0
content_weight = 100.0
noise_weight = 1.0

num_training_steps = num_epochs * len(dataset) // batch_size
progress_bar = tqdm(range(num_training_steps), desc="Fine-tuning Vision Model")
lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
# lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

for epoch in range(num_epochs):
    # 为当前epoch分配图像对
    pair_assignments = pair_assigner.assign_pairs()
    dataloader = create_epoch_dataloader(dataset, batch_size, pair_assignments)
    
    epoch_loss = 0.0
    epoch_noise_loss = 0.0
    epoch_style_loss = 0.0
    epoch_content_loss = 0.0
    
    for batch in dataloader:
        images = batch["image"].to(device)
        other_images = batch["other_image"].to(device)
        captions = batch["caption"]
        
        # Tokenize captions
        tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer")
        inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pos = inputs["attention_mask"].sum(dim=1) - 3
        
        # 1. 原始图像的attribute vector
        attribute_vector = vision_mapping_model(images)
        
        # 2. 另一张图像的attribute vector
        other_attribute_vector = vision_mapping_model(other_images)
        
        # 3. 使用原始attribute vector构建条件，计算noise loss
        final_embeddings = get_final_embeddings(text_encoder, inputs["input_ids"],
                                                attribute_vector=attribute_vector, pos=pos)
        final_embeddings = text_encoder.text_model(input_ids=inputs["input_ids"] ,hidden_states=final_embeddings)[0]

        # 获取原始图像的latent表示
        with torch.no_grad():
            latent_dist = vae.encode(images).latent_dist
            latents = latent_dist.sample() * vae.config.scaling_factor
        
        # 添加噪声并计算noise loss
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=latents.device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        
        # 预测噪声
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_embeddings, return_dict=False)[0]
        noise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        # 4. 使用other_attribute_vector和caption生成新图像，计算style loss和content loss
        other_embeddings = get_final_embeddings(text_encoder, inputs["input_ids"],
                                              attribute_vector=other_attribute_vector, pos=pos)
        other_embeddings = text_encoder.text_model(input_ids=inputs["input_ids"], hidden_states=other_embeddings)[0]
        
        # 生成一个简化的图像latent表示（使用diffusion模型的部分步骤）
        generated_latents = diffusion_generation_step(unet, noise, timesteps, other_embeddings)
        
        # 计算style loss
        style_loss = calculate_style_loss(generated_latents, other_images, vision_mapping_model, vae)
        
        # 计算content loss
        content_loss = calculate_content_loss(generated_latents, images, clip_model, vae)
        
        # 5. 总损失 = noise_loss + style_weight * style_loss + content_weight * content_loss
        total_loss = (noise_weight * noise_loss + 
                      style_weight * style_loss + 
                      content_weight * content_loss)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # 累计损失
        epoch_loss += total_loss.item()
        epoch_noise_loss += noise_loss.item()
        epoch_style_loss += style_loss.item()
        epoch_content_loss += content_loss.item()
        
        # 记录到wandb
        wandb.log({
            "batch_total_loss": total_loss.item(),
            "batch_noise_loss": noise_loss.item(),
            "batch_style_loss": style_loss.item(),
            "batch_content_loss": content_loss.item(),
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })
        
        progress_bar.update(1)
        progress_bar.set_postfix(epoch=epoch+1, loss=total_loss.detach().item(), noise_loss=noise_loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
    
    # 计算并记录每个epoch的平均损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_epoch_noise_loss = epoch_noise_loss / len(dataloader)
    avg_epoch_style_loss = epoch_style_loss / len(dataloader)
    avg_epoch_content_loss = epoch_content_loss / len(dataloader)
    
    wandb.log({
        "epoch": epoch + 1,
        "epoch_total_loss": avg_epoch_loss,
        "epoch_noise_loss": avg_epoch_noise_loss,
        "epoch_style_loss": avg_epoch_style_loss,
        "epoch_content_loss": avg_epoch_content_loss
    })
    print(f"Epoch {epoch+1}/{num_epochs} finished. Average total loss: {avg_epoch_loss:.4f}")
    
    # 保存精调后的模型
    save_path = os.path.join(model_dir, f"vision_mapping_model_new_epoch{epoch+1}.pt")
    torch.save(vision_mapping_model.state_dict(), save_path)
    print(f"Saved fine-tuned vision mapping model to {save_path}")
    

# 结束wandb运行
wandb.finish()