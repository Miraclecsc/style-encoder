import torch
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DiffusionPipeline
from PIL import Image
import os
import numpy as np
import json
import random
from my_clip.modeling_clip import CLIPTextModel

# Paths to the models and saved attribute vectors
stable_diffusion_path = "/data2/changshuochen/model/stable-diffusion-v1-4"
clip_path = "/data2/changshuochen/model/clip-vit-large-patch14-local"
attribute_vector_path = "/data2/changshuochen/model/style-embedding"
device = "cuda:3"  # Use a single GPU (cuda:0)

# Load the models
text_encoder = CLIPTextModel.from_pretrained(clip_path).to(device)
vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(stable_diffusion_path, subfolder="unet").to(device)
noise_scheduler = PNDMScheduler.from_pretrained(stable_diffusion_path, subfolder="scheduler")
tokenizer = AutoTokenizer.from_pretrained(stable_diffusion_path, subfolder="tokenizer")
pipeline  = DiffusionPipeline.from_pretrained(stable_diffusion_path)
pipeline.safety_checker = None
pipeline = pipeline.to(device)

# Function to get final embeddings with the attribute vector
def get_final_embeddings(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    original_token_embeddings = token_embed(input_ids)  # [batch, seq_len, hidden_dim]
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])  # [1, seq_len, hidden_dim]

    if attribute_vector is not None and pos is not None:
        original_token_embeddings[:, pos, :] = attribute_vector

    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

def get_final_embeddings_single(_text_encoder, input_ids, attribute_vector=None, pos=None):
    embeddings_layer = _text_encoder.text_model.embeddings
    token_embed = embeddings_layer.token_embedding
    pos_embed = embeddings_layer.position_embedding
    pos_ids = embeddings_layer.position_ids

    original_token_embeddings = token_embed(input_ids)  # [batch, seq_len, hidden_dim]
    print(original_token_embeddings.shape)
    position_embeddings = pos_embed(pos_ids[:, :input_ids.shape[1]])  # [1, seq_len, hidden_dim]
    print(position_embeddings.shape)

    if attribute_vector is not None and pos is not None:
        # for i in range(len(pos)):
        pos_sks = pos
        end_embeddings = original_token_embeddings[:, pos_sks + 1: pos_sks + 3, :]
        original_token_embeddings[:, pos_sks: pos_sks + 8, :] = attribute_vector
        original_token_embeddings[:, pos_sks + 8: pos_sks + 10, :] = end_embeddings

    final_embeddings = original_token_embeddings + position_embeddings
    return final_embeddings

# json_path = "/home/changshuochen/model/image_captions_shuffle.json"
json_path = "/data2/changshuochen/model/style30k/image_captions.json"
with open(json_path, 'r') as f:
    captions_data = json.load(f)
# 创建image_path到caption的字典
caption_dict = {item["image_path"]: item["caption"] for item in captions_data}

# 修改attribute_vectors的加载逻辑
attribute_vectors = {}
for file_name in os.listdir(attribute_vector_path):
    if file_name.endswith(".pt"):
        # 正确解析image_path（去掉"attribute_"前缀和"_step_"后缀）
        image_part = file_name[len("attribute_"):]  # 移除开头的"attribute_"
        image_path = image_part.split("_step_")[0]  # 分割出image_path部分
        attribute_vectors[image_path] = torch.load(os.path.join(attribute_vector_path, file_name)).to(device)

# Test function to generate images
def generate_image(text_prompt, attribute_vector, pt_filename, height=512, width=512):
    # Tokenize the input text
    inputs = tokenizer(text_prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)

    # Use the attribute vector to modify the style
    final_embeddings = get_final_embeddings_single(text_encoder, inputs["input_ids"], attribute_vector=attribute_vector, pos=inputs["attention_mask"].sum() - 3)

    # Forward pass through the text encoder
    output = text_encoder.text_model(input_ids=inputs["input_ids"], hidden_states=final_embeddings)
    encoder_hidden_states = output[0]
    generated_images = pipeline(prompt_embeds=encoder_hidden_states, num_inference_steps=50, guidance_scale=7.5).images
    
    # 创建输出目录
    # output_dir = "/home/changshuochen/model/generated_images_shuffle"
    output_dir = "/data2/changshuochen/model/generated_images_emb"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 .pt 文件名作为基础，将扩展名改为 .png
    base_name = os.path.splitext(pt_filename)[0]
    # base_name += text_prompt.replace(" ", "_")
    output_path = os.path.join(output_dir, f"{base_name}.png")
    for idx, img in enumerate(generated_images):
        img.save(output_path)
        print(f"图片已生成: {output_path}")


pt_files = [f for f in os.listdir(attribute_vector_path) if f.endswith(".pt")]

# 创建文件对的字典
file_pairs = {}
for pt_file in pt_files:
    # 获取基础名称（不包含step部分）
    base_name = pt_file.rsplit('_step_', 1)[0]
    if base_name not in file_pairs:
        file_pairs[base_name] = []
    file_pairs[base_name].append(pt_file)

# 获取所有基础名称并随机打乱
base_names = list(file_pairs.keys())
random.shuffle(base_names)

# 重新构建文件列表，保持对子在一起
pt_files = []
for base_name in base_names:
    pt_files.extend(file_pairs[base_name])

total_files = len(pt_files)

for idx, pt_filename in enumerate(pt_files, 1):
    print(f"处理进度: [{idx}/{total_files}] - {pt_filename}")
    
    # 解析当前.pt文件对应的image_path
    image_part = pt_filename[len("attribute_"):]
    image_path = image_part.split("_step_")[0]
    
    # 从字典获取对应的caption
    if image_path not in caption_dict:
        print(f"警告：未找到{image_path}对应的caption，跳过此文件")
        continue
    
    if image_path not in attribute_vectors:
        print(f"警告：未找到{image_path}对应的属性向量，跳过此文件")
        continue
    
    text_prompt = caption_dict[image_path]
    attribute_vector = attribute_vectors[image_path]
    
    # 生成图片（移除原函数中硬编码的text_prompt）
    generate_image(text_prompt, attribute_vector, pt_filename)
    
print(f"所有图片生成完成！共处理了 {total_files} 个文件。")