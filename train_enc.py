import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPModel
from PIL import Image
import json
import os
from torchvision import transforms
from transformers import CLIPProcessor
import wandb  # 导入 wandb

# Define paths
clip_path = "/data2/changshuochen/model/clip-vit-large-patch14-local"
device = "cuda:0"

# 目标 attribute_vector 存储目录与训练步数
attribute_dir = "/data2/changshuochen/model/style-embedding"
model_dir = "/data2/changshuochen/model/encoder-models"
os.makedirs(model_dir, exist_ok=True)
target_step = 200  # 默认选取保存 step200 的向量

# 训练用图片数据所在目录及数据列表
image_data_root = "/data2/changshuochen/model/style30k/images"
json_path = "/data2/changshuochen/model/style30k/image_captions.json"

# 训练超参数
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
eps = 1e-8

# 定义图像预处理
image_size = 224
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 其余的 VisionMappingDataset 和 VisionMappingModel 类保持不变...

class VisionMappingDataset(Dataset):
    """
    数据集：每个样本返回一张图片及其对应的 attribute_vector（8,768）。
    attribute_vector 文件路径格式：attribute_{image_filename}_step_{target_step}.pt
    """
    def __init__(self, json_path, image_data_root, attribute_dir, transform=None, target_step=200):
        super().__init__()
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.image_data_root = image_data_root
        self.attribute_dir = attribute_dir
        self.transform = transform
        self.target_step = target_step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_rel_path = sample["image_path"]
        image_path = os.path.join(self.image_data_root, image_rel_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        # 构造 attribute_vector 文件路径
        image_filename = os.path.basename(image_rel_path)
        attribute_file = os.path.join(self.attribute_dir, f"attribute_{image_filename}_step_{self.target_step}.pt")
        if not os.path.exists(attribute_file):
            raise FileNotFoundError(f"Attribute vector file not found: {attribute_file}")
        target = torch.load(attribute_file, weights_only=True)  # 形状应为 [8,768]
        
        return {
            "image": image,
            "target": target,  # target tensor shape: [8,768]
            "image_path": image_path
        }

# 定义映射模型，在 CLIP vision model 基础上加一个 projection head
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


def main():
    # 初始化 wandb
    wandb.init(
        project="style-encoder",  # 项目名称
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "model": "VisionMappingModel",
            "target_step": target_step,
            "clip_model": clip_path,
        }
    )
    
    # 构建数据集与 DataLoader
    dataset = VisionMappingDataset(
        json_path=json_path,
        image_data_root=image_data_root,
        attribute_dir=attribute_dir,
        transform=transform,
        target_step=target_step
    )
    
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        image_paths = [item["image_path"] for item in batch]
        return {"image": images, "target": targets, "image_path": image_paths}
    
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    # 加载 CLIP 模型，并构建映射模型
    clip_model = CLIPModel.from_pretrained(clip_path)
    mapping_model = VisionMappingModel(clip_model).to(device)
    
    # wandb 可以记录模型架构
    wandb.watch(mapping_model)
    
    optimizer = AdamW(mapping_model.parameters(), lr=learning_rate, eps=eps)
    progress_bar = tqdm(range(num_epochs * len(dataloader)), desc="Training Epochs")

    mapping_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            outputs = mapping_model(images)
            loss = F.mse_loss(outputs, targets, reduction="mean")
            loss.backward()
            optimizer.step()

            # 记录每一步的损失，添加 global_step 和 epoch 信息
            global_step = epoch * len(dataloader) + step
            wandb.log({
                "batch_loss": loss.item(),
                "epoch": epoch,
                "batch": step,
                "global_step": global_step
            })
            epoch_loss += loss.item()
            
            progress_bar.update(1)
            progress_bar.set_postfix(epoch=epoch, loss=loss.detach().item())
        
        # 记录每个 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({
            "epoch": epoch, 
            "epoch_loss": avg_epoch_loss
        })
        ckpt_path = os.path.join(model_dir, f"vision_mapping_model_epoch{epoch}.pt")
        torch.save(mapping_model.state_dict(), ckpt_path)
        print(f"Epoch {epoch+1}/{num_epochs} finished. Avg loss: {avg_epoch_loss:.6f}")

    # 保存训练后的模型
    model_path = os.path.join(model_dir, "vision_mapping_model.pt")
    torch.save(mapping_model.state_dict(), model_path)
    
    # 将模型上传到 wandb
    wandb.save(model_path)
    
    # 完成训练，关闭 wandb
    wandb.finish()

if __name__ == '__main__':
    main()