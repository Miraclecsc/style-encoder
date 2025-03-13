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

device = "cuda:0"

# 路径设置
image_data_root = '/data2/changshuochen/model/style30k/images'
prompts_root = '/data2/changshuochen/model/style30k/new.json'
clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
stable_diffusion_path = '/data2/changshuochen/model/stable-diffusion-v1-4'
model_dir = '/data2/changshuochen/model/encoder-models'
vision_mapping_model_path = os.path.join(model_dir, "vision_mapping_model_finetuned_epoch5.pt")

# 图像预处理（保持和训练时一致）
image_size = 224  # 可根据需要调整
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 定义 get_final_embeddings 函数（用于拼接 attribute vector）
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
        for i in range(batch_size):
            start = pos[i].item() if torch.is_tensor(pos[i]) else pos[i]
            # 保存原本 start+1:start+3（如句号和 eos）的 embedding
            end_embeddings = original_token_embeddings[i, start + 1: start + 3, :].clone()
            # 将从 start 开始的连续 8 个 token 替换为 attribute_vector（注意：attribute_vector[i] shape 为 [8,hidden_size]）
            original_token_embeddings[i, start: start + 8, :] = attribute_vector[i]
            # 将原来的标点放到 pos+8:pos+10
            original_token_embeddings[i, start + 8: start + 10, :] = end_embeddings
    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

# 定义 VisionMappingModel，将 CLIP 视觉部分映射到 8*hidden_size 的向量
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

# 辅助函数：从 JSON 数据中获取单个样本的 image 和 caption
with open(prompts_root, "r") as f:
    captions_data = json.load(f)
    # 打乱顺序
    random.shuffle(captions_data)

def get_sample(idx):
    sample = captions_data[idx]
    # 假设 JSON 中包含 "image_path" 和 "caption" 字段
    image_path = os.path.join(image_data_root, sample["image_path"])
    caption = sample["caption"]
    pth = sample["image_path"]
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image, caption, pth

def main():
    # 加载 tokenizer 和 text encoder（CLIP 的 text encoder）
    tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(clip_path).to(device)
    
    # 加载 CLIP 模型，用于构造 VisionMappingModel
    from transformers import CLIPModel
    clip_model = CLIPModel.from_pretrained(clip_path)
    vision_mapping_model = VisionMappingModel(clip_model).to(device)
    vision_mapping_model.load_state_dict(torch.load(vision_mapping_model_path, map_location=device))
    vision_mapping_model.eval()
    
    # 加载 Stable Diffusion pipeline（注意：传入 safety_checker=None 避免不必要的警告）
    pipeline = StableDiffusionPipeline.from_pretrained(stable_diffusion_path, safety_checker=None).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # 测试若干个样本，这里以前 5 个为例
    num_test_samples = 500
    output_dir = "/data2/changshuochen/model/generated_images_simple"
    os.makedirs(output_dir, exist_ok=True)
    for idx in tqdm(range(num_test_samples), desc="Testing Samples"):
        image, caption, pth = get_sample(idx)
        # 扩展 batch 维度并送入 GPU
        image_tensor = image.unsqueeze(0).to(device)  # shape: [1, C, H, W]
        
        # 通过 vision mapping model 得到 attribute vector（输出 shape: [1, 8, hidden_size]）
        with torch.no_grad():
            attribute_vector = vision_mapping_model(image_tensor)
        
        # 对 caption 进行 tokenize
        inputs = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        # 将所有 tensor 移动到 device 上
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # 计算替换位置：按照之前训练时的逻辑，pos = attention_mask.sum() - 3
        pos = inputs["attention_mask"].sum(dim=1) - 3  # shape: [1]
        
        # 拼接 attribute vector 到文本 embedding 中
        final_embeddings = get_final_embeddings(text_encoder, inputs["input_ids"], attribute_vector=attribute_vector, pos=pos)
        
        # 用修改后的 embedding 进行前向，获取 encoder_hidden_states
        output = text_encoder.text_model(input_ids=inputs["input_ids"], hidden_states=final_embeddings)
        encoder_hidden_states = output[0]
        
        # 通过 diffusion pipeline 生成图像（传入 prompt_embeds 覆盖原有文本 embedding）
        generated = pipeline(prompt_embeds=encoder_hidden_states, num_inference_steps=50, guidance_scale=7.5)
        generated_images = generated.images
        
        # 保存生成的图像
        save_path = os.path.join(output_dir, pth)
        generated_images[0].save(save_path)
        print(f"Saved generated image to {save_path}")

if __name__ == '__main__':
    main()