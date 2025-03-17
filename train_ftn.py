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

# -------------------------------
# 1. 路径和超参数设置
# -------------------------------
image_data_root = '/data2/changshuochen/model/style30k/images'
prompts_root = '/data2/changshuochen/model/style30k/image_captions.json'
model_dir = '/data2/changshuochen/model/encoder-models'
vision_mapping_model_path = os.path.join(model_dir, "vision_mapping_model.pt")
stable_diffusion_path = '/data2/changshuochen/model/stable-diffusion-v1-4'
clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
device = "cuda:3"

batch_size = 8
num_epochs = 5
learning_rate = 2e-4

# 初始化wandb
wandb.init(project="style-encoder-finetuning", 
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
            # 返回一个随机噪声图像作为替代
            image = torch.randn(3, image_size, image_size)
        return {"image": image, "caption": caption, "image_path": image_path}

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    return {"image": images, "caption": captions, "image_path": image_paths}

dataset = ImageCaptionDataset(image_data_root, prompts_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# -------------------------------
# 3. 加载模型（文本 encoder、VAE、UNet、scheduler均冻结）
# -------------------------------
# 文本 encoder（CLIP Text Model）
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

from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained(clip_path)
vision_mapping_model = VisionMappingModel(clip_model).to(device)
vision_mapping_model.load_state_dict(torch.load(vision_mapping_model_path, map_location=device))
# 精调时将 vision mapping model 置为训练模式
vision_mapping_model.train()

optimizer = AdamW(vision_mapping_model.parameters(), lr=learning_rate, eps=1e-8)
# 添加学习率调度器
lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))
# lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

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
num_training_steps = num_epochs * len(dataloader)
progress_bar = tqdm(range(num_training_steps), desc="Fine-tuning Vision Model")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device)  # [B, C, H, W]
        captions = batch["caption"]
        # Tokenize captions
        tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer")
        inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # 计算替换位置：pos = attention_mask.sum() - 3（每个样本）
        pos = inputs["attention_mask"].sum(dim=1) - 3  # [B]

        # 用 vision mapping model 得到 attribute vector（风格条件）
        attribute_vector = vision_mapping_model(images)  # [B, 8, hidden_size]

        # 注入 attribute vector 得到最终文本 embedding（作为条件）
        final_embeddings = get_final_embeddings(text_encoder, inputs["input_ids"],
                                                attribute_vector=attribute_vector, pos=pos)
        # encoder_hidden_states = text_encoder.text_model(input_ids=inputs["input_ids"] ,hidden_states=final_embeddings)[0]

        # 使用 VAE 将图像编码为 latent 表示
        with torch.no_grad():
            latent_dist = vae.encode(images).latent_dist
            latents = latent_dist.sample() * vae.config.scaling_factor

        # 随机采样噪声
        noise = torch.randn_like(latents)
        # 随机采样 timesteps
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.shape[0],), device=latents.device).long()
        # 向 latent 中添加噪声
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # 利用 UNet 在给定条件下预测噪声
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_embeddings, return_dict=False)[0]

        # 计算 MSE loss（标准噪声 loss）
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # 累计epoch损失
        epoch_loss += loss.item()
        
        # 记录到wandb
        wandb.log({
            "batch_loss": loss.item(),
            "learning_rate": lr_scheduler.get_last_lr()[0]
        })

        progress_bar.update(1)
        progress_bar.set_postfix(epoch=epoch+1, loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
    
    # 计算并记录每个epoch的平均损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": avg_epoch_loss
    })
    print(f"Epoch {epoch+1}/{num_epochs} finished. Average loss: {avg_epoch_loss:.4f}")

    # -------------------------------
    # 7. 保存精调后的模型
    # -------------------------------
    save_path = os.path.join(model_dir, f"vision_mapping_model_finetuned_epoch{epoch+1}.pt")
    torch.save(vision_mapping_model.state_dict(), save_path)
    print(f"Saved fine-tuned vision mapping model to {save_path}")

# 结束wandb运行
wandb.finish()