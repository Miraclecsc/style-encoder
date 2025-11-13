from transformers import CLIPModel
import torch
from diffusers import AutoencoderKL

clip_path = '/data2/changshuochen/model/clip-vit-large-patch14-local'
stable_diffusion_path = '/data2/changshuochen/model/stable-diffusion-v1-4'
device = "cuda:0"
image_size = 224

clip_model = CLIPModel.from_pretrained(clip_path).to(device)
vae = AutoencoderKL.from_pretrained(stable_diffusion_path, subfolder="vae").to(device)

with torch.no_grad():
    test_image = torch.randn(4, 3, image_size, image_size).to(device)
    test_latent = torch.randn(1, 4, 64, 64).to(device)  # 随机latent
    output_images = vae.decode(test_latent).sample      # 解码
    print("Min value:", output_images.min().item())
    print("Max value:", output_images.max().item())

    # 方法1：直接调用 encode_image
    encode_image_out = clip_model.get_image_features(test_image)

    # 方法2：手动组合 vision_model + visual_projection
    vision_outputs = clip_model.vision_model(test_image)
    pooler_output = vision_outputs.pooler_output
    projected_output = clip_model.visual_projection(pooler_output)

    print("encode_image_out.shape:", encode_image_out.shape)
    print("pooler_output.shape:", pooler_output.shape)
    print("projected_output.shape:", projected_output.shape)

    # 使用allclose判断数值是否相近
    is_close = torch.allclose(encode_image_out, projected_output, atol=1e-7)
    print("手动组合与encode_image是否数值接近:", is_close)