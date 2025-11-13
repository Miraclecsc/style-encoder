from transformers import CLIPModel
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

clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
device = "cuda:0"

clip_model = CLIPModel.from_pretrained(clip_path)
clip_model = clip_model.to(device)
image_size = 224
# 测试CLIP模型的调用方式
with torch.no_grad():
    test_image = torch.randn(4, 3, image_size, image_size).to(device)
    try:
        features = clip_model.encode_image(test_image)
        print("成功使用clip_model.encode_image()")
    except Exception as e:
        print(f"encode_image失败: {e}")
        
    try:
        features = clip_model.get_image_features(pixel_values=test_image)
        print("成功使用clip_model.get_image_features(pixel_values=...)")
    except Exception as e:
        print(f"get_image_features失败: {e}")
        
    try:
        features1 = clip_model.vision_model(test_image).pooler_output
        print("成功使用clip_model.vision_model(...).pooler_output")
    except Exception as e:
        print(f"vision_model失败: {e}")
        
    print(features == features1)