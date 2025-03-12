import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from my_clip.modeling_clip import CLIPTextModel
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np

from transformers import AutoTokenizer
from my_clip.modeling_clip import CLIPTextModel
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel

device = "cuda:0"

# ================================
# 1. 路径及超参数设置
# ================================
image_data_root = '/data2/changshuochen/model/style30k/images'
prompts_root = '/data2/changshuochen/model/style30k/image_captions.json'
clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
stable_diffusion_path = '/data2/changshuochen/model/stable-diffusion-v1-4'
vision_mapping_model_path = os.path.join('/data2/changshuochen/model/encoder-models', "vision_mapping_model.pt")

num_inference_steps = 50
height = 224
width = 224
# 控制文本与风格条件的权重
st = 7.5  # 文本条件权重
ss = 1.0  # 风格条件权重

# ================================
# 2. 图像预处理
# ================================
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.CenterCrop((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

# ================================
# 3. 数据加载辅助函数
# ================================
with open(prompts_root, "r") as f:
    captions_data = json.load(f)
    random.shuffle(captions_data)

def get_sample(idx):
    """
    返回指定索引的样本，包括图像和 caption
    """
    sample = captions_data[idx]
    image_path = os.path.join(image_data_root, sample["image_path"])
    caption = sample["caption"]
    pth = sample["image_path"]
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image, caption, pth

# ================================
# 4. 定义 get_final_embeddings 函数
#    用于将属性向量（风格条件）拼接到文本 embedding 上
# ================================
def get_final_embeddings(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    original_token_embeddings = token_embed(input_ids)  # [B, seq_len, hidden_size]
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])
    if attribute_vector is not None and pos is not None:
        batch_size, num_attr, hidden_size = attribute_vector.shape
        for i in range(batch_size):
            start = pos[i].item() if torch.is_tensor(pos[i]) else pos[i]
            # 保存原本 start+1:start+3 的标点（如句号和 eos）的 embedding
            end_embeddings = original_token_embeddings[i, start + 1: start + 3, :].clone()
            # 用属性向量替换从 start 开始的连续 8 个 token
            original_token_embeddings[i, start: start + 8, :] = attribute_vector[i]
            # 将原来的标点放到后面：start+8:start+10
            original_token_embeddings[i, start + 8: start + 10, :] = end_embeddings
    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

# ================================
# 5. 加载各模型
# ================================
# 文本相关模型
tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(clip_path).to(device)

# 扩散模型相关
unet = UNet2DConditionModel.from_pretrained(stable_diffusion_path, subfolder="unet").to(device)
vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)
scheduler = PNDMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")

# ================================
# 6. 加载 vision mapping model（用于将图像映射到属性向量，形状 [8, hidden_size]）
# ================================
# 此处构造与训练时相同的模型结构
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

from transformers import CLIPModel
clip_model = CLIPModel.from_pretrained(clip_path)
vision_mapping_model = VisionMappingModel(clip_model).to(device)
vision_mapping_model.load_state_dict(torch.load(vision_mapping_model_path, map_location=device))
vision_mapping_model.eval()

# ================================
# 7. 构造三组文本 embedding：
#    (1) 无条件：eθ(zt, ∅, ∅)
#    (2) 仅文本条件：eθ(zt, ct, ∅)
#    (3) 文本+风格条件：eθ(zt, ct, cs)
# ================================
# (1) 无条件 embedding（用空文本生成）
uncond_inputs = tokenizer("", padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
with torch.no_grad():
    embedding_uncond = text_encoder.text_model(**uncond_inputs)[0]  # [1, seq_len, hidden_size]

def get_text_only_embedding(caption):
    #把caption末尾的", in the style of sks."去掉
    caption = caption.split(", in the style of sks.")[0]
    caption += "."
    inputs = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = text_encoder.text_model(**inputs)[0]  # [1, seq_len, hidden_size]
    return inputs, embedding

def get_full_embedding(caption, attribute_vector):
    inputs = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    # 计算替换位置：例如 pos = attention_mask.sum() - 3
    pos = inputs["attention_mask"].sum(dim=1) - 3  # [1]
    full_embedding = get_final_embeddings(text_encoder, inputs["input_ids"], attribute_vector=attribute_vector, pos=pos)
    return inputs, full_embedding

# ================================
# 8. 定义自定义扩散去噪过程，采用双重指导
# ================================
def run_diffusion(embedding_uncond, embedding_text_only, embedding_full):
    batch_size = 1
    # 初始化随机噪声
    latents = torch.randn((batch_size, unet.in_channels, 64, 64), device=device)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        with torch.no_grad():
            # 分别预测三种条件下的噪声
            noise_pred_uncond = unet(latents, t, encoder_hidden_states=embedding_uncond).sample
            noise_pred_text = unet(latents, t, encoder_hidden_states=embedding_text_only).sample
            noise_pred_full = unet(latents, t, encoder_hidden_states=embedding_full).sample

            # 根据公式组合：
            # noise_pred = noise_pred_uncond + st*(noise_pred_text - noise_pred_uncond) + ss*(noise_pred_full - noise_pred_text)
            noise_pred = noise_pred_uncond + \
                         st * (noise_pred_text - noise_pred_uncond) + \
                         ss * (noise_pred_full - noise_pred_text)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # 使用 VAE 解码为图像
    with torch.no_grad():
        latents = 1.0 / vae.config.scaling_factor * latents
        image_tensor = vae.decode(latents).sample
    # 后处理：调整到 [0,255]并转换为 PIL Image
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_np = image_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
    image_np = (image_np * 255).astype(np.uint8)
    image = Image.fromarray(image_np)
    return image

# ================================
# 9. 主流程：对每个样本分别生成图像
# ================================
def main():
    num_test_samples = 500
    output_dir = "/data2/changshuochen/model/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    for idx in tqdm(range(num_test_samples), desc="Testing Samples"):
        # 获取图片与 caption
        image, caption, pth = get_sample(idx)
        image_tensor = image.unsqueeze(0).to(device)  # [1, C, H, W]

        # 通过 vision mapping model 得到风格属性向量（cs）
        with torch.no_grad():
            attribute_vector = vision_mapping_model(image_tensor)  # [1, 8, hidden_size]

        # 获取仅文本条件下的 embedding
        _, embedding_text_only = get_text_only_embedding(caption)
        # 获取文本+风格条件下的 embedding
        _, embedding_full = get_full_embedding(caption, attribute_vector)

        # 运行自定义扩散生成过程
        generated_image = run_diffusion(embedding_uncond, embedding_text_only, embedding_full)

        # 保存生成结果
        save_path = os.path.join(output_dir, pth)
        generated_image.save(save_path)
        print(f"Saved generated image to {save_path}")

if __name__ == '__main__':
    main()