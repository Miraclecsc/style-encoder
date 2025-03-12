import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from my_clip.modeling_clip import CLIPTextModel
from PIL import Image
import json
import os
import diffusers
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from torchvision import transforms

# Define paths
stable_diffusion_path = "/data2/changshuochen/model/stable-diffusion-v1-4"
clip_path = "/data2/changshuochen/model/clip-vit-large-patch14-local"
device = "cuda:0"  # Use a single GPU (cuda:0)

# Function to get final embeddings with the attribute vector
def get_final_embeddings_single(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    original_token_embeddings = token_embed(input_ids)  # [batch, seq_len, hidden_dim]
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])  # [1, seq_len, hidden_dim]

    if attribute_vector is not None and pos is not None:
        for i in range(len(pos)):
            pos_sks = pos[i]
            end_embeddings = original_token_embeddings[i, pos_sks + 1: pos_sks + 3, :]
            original_token_embeddings[i, pos_sks: pos_sks + 8, :] = attribute_vector
            original_token_embeddings[i, pos_sks + 8: pos_sks + 10, :] = end_embeddings

    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

def get_final_embeddings(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    # 原始 token embeddings: [B, seq_len, hidden_dim]
    original_token_embeddings = token_embed(input_ids)
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])

    if attribute_vector is not None and pos is not None:
        batch_size, num_attr, hidden_size = attribute_vector.shape
        # 对每个样本进行替换：用 attribute_vector 替换从 pos 到 pos+num_attr 的位置，
        # 并将原本 pos+num_attr 后的标点（如句号、eos）接到后面
        for i in range(batch_size):
            start = pos[i].item() if torch.is_tensor(pos[i]) else pos[i]
            # 假设原来的设计是：原本在 start 位置为 "sks"，start+1:start+3 为标点，
            # 用 attribute_vector 替换 start:start+8，然后把原来 start+1:start+3 的标点放到 start+8:start+10
            end_embeddings = original_token_embeddings[i, start + 1: start + 3, :].clone()
            original_token_embeddings[i, start: start + 8, :] = attribute_vector[i]
            original_token_embeddings[i, start + 8: start + 10, :] = end_embeddings

    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

class StyleDataset(Dataset):
    def __init__(self, image_data_root, prompts_root, tokenizer, size=512, tokenizer_max_length=77, center_crop=True):
        self.image_data_root = image_data_root
        self.prompts_root = prompts_root
        self.tokenizer = tokenizer
        self.size = size
        self.tokenizer_max_length = tokenizer_max_length
        self.center_crop = center_crop

        with open(prompts_root, "r") as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size) if self.center_crop else transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.attribute_vectors = {}
        hidden_size = 768
        num_tokens = 8
        for image_info in self.data:
            image_path = image_info["image_path"]
            self.attribute_vectors[image_path] = nn.Parameter(torch.zeros(num_tokens, hidden_size, device=device))  # Move to GPU
            nn.init.normal_(self.attribute_vectors[image_path].data, mean=0.0, std=0.02)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image_path = os.path.join(self.image_data_root, image_info["image_path"])
        caption = image_info["caption"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        inputs = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt")

        # Return the attribute_vector with requires_grad=True
        attribute_vector = self.attribute_vectors[image_info["image_path"]]

        return {
            "images": image,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "text_prompts": caption,
            "pos": inputs["attention_mask"].squeeze(0).sum() - 3,
            "image_path": image_path,
            "image_ids": image_info["image_path"],
            "attribute_vector": attribute_vector  # No need for detach() here
        }

def train(dataset, batch_size, max_train_steps_per_image, global_step):
    # Set device to cuda0 (single GPU)
    torch.cuda.set_device(device)

    # Initialize models and move to device
    text_encoder = CLIPTextModel.from_pretrained(clip_path).to(device)
    vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(stable_diffusion_path, subfolder="unet").to(device)
    noise_scheduler = PNDMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

    # Freeze the parameters of the models
    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in unet.parameters():
        param.requires_grad = False

    # Prepare the dataset (no need for DistributedSampler, single GPU)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # This batch size will be used for the single GPU
    )

    # Create optimizer only for the attribute vectors
    optimizer = AdamW([param for param in dataset.attribute_vectors.values() if param.requires_grad], lr=1e-3, eps=1e-8)

    # progress_bar = tqdm(range(len(dataset) * max_train_steps_per_image), desc="Training Steps")
    progress_bar = tqdm(range(len(dataloader) * max_train_steps_per_image), desc="Training Steps")
    
    output_dir = "/data2/changshuochen/model/style-embedding"
    resume_mode = True  # 初始状态下开启断点检查

    # Training loop

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        batch_pos = batch["pos"].to(device)  # 确保 pos 也是 tensor
        
        if resume_mode:
            # 检查当前 batch 中所有图像是否已经存在最终 checkpoint 文件
            skip_batch = True
            for image_path in batch["image_path"]:
                filename = f"attribute_{os.path.basename(image_path)}_step_{max_train_steps_per_image}.pt"
                output_file = os.path.join(output_dir, filename)
                if not os.path.exists(output_file):
                    skip_batch = False
                    break
            if skip_batch:
                # print(f"Skipping batch {batch_idx} because final checkpoint exists")
                progress_bar.update(200)
                continue
            else:
                # 一旦有一个 batch 没有全都有 checkpoint，就关闭后续检查
                resume_mode = False   
                print(f"Resuming training from batch {batch_idx}")         

        # 将 attribute vector 批次化
        batch_attribute_vectors = torch.stack(
            [dataset.attribute_vectors[image_ids] for image_ids in batch["image_ids"]],
            dim=0
        )

        for step in range(max_train_steps_per_image):
            # 注意这里的 batch_loss 要用 0.0 而不是每次新建 requires_grad=True 的 tensor，
            # 可以直接将 loss 累加后再调用 backward
            batch_loss = 0.0

            # VAE 编码可以并行执行
            with torch.no_grad():
                model_input = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(model_input)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=model_input.device).long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # 得到整个 batch 的 embedding
            final_embeddings = get_final_embeddings(text_encoder, input_ids, attribute_vector=batch_attribute_vectors, pos=batch_pos)
            output = text_encoder.text_model(input_ids=input_ids, hidden_states=final_embeddings)
            encoder_hidden_states = output[0]

            model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
            target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            batch_loss += loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            global_step += 1
            progress_bar.update(1)

            if (step + 1) % 50 == 0:
                # 保存每个图像对应的 attribute vector
                for i, image_path in enumerate(batch["image_path"]):
                    attr_copy = batch_attribute_vectors[i].detach().cpu().clone()
                    torch.save(attr_copy, os.path.join("/data2/changshuochen/model/style-embedding", f"attribute_{image_path.split('/')[-1]}_step_{step + 1}.pt"))

        print(f"Completed training for batch {batch_idx + 1}")

def main():
    dataset = StyleDataset(
        image_data_root='/data2/changshuochen/model/style30k/new_splits/split_3/images', 
        prompts_root='/data2/changshuochen/model/style30k/new_splits/split_3/captions_3.json',
        tokenizer=AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer"),
        size=512,
        tokenizer_max_length=77
    ) 
    batch_size = 8  # Adjust based on your GPU capacity
    max_train_steps_per_image = 200  # Training each image for 500 steps
    global_step = 0

    train(dataset, batch_size, max_train_steps_per_image, global_step)

if __name__ == '__main__':
    main()