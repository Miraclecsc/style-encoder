import os
from PIL import Image


# 原始图片和重构图片的路径
original_images_dir = '/data2/changshuochen/model/style30k/images'
generated_images_dir = '/data2/changshuochen/model/generated_images_emb'
output_dir = '/data2/changshuochen/model/output_images_emb'  # 输出的图像目录

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历原始图片文件夹
for filename in os.listdir(original_images_dir):
    try:
        if filename.endswith('.jpg'):
            # 原始图像的完整路径
            original_image_path = os.path.join(original_images_dir, filename)

            # 查找对应的重构图像
            base_filename = filename.split('.jpg')[0]  # 提取文件名（去掉.jpg后缀）
            reconstructed_images = []

            # 查找step_*格式的图片
            for step_filename in os.listdir(generated_images_dir):
                if base_filename in step_filename and step_filename.endswith('.png'):
                    reconstructed_images.append(os.path.join(generated_images_dir, step_filename))

            # 如果找到了两张重构图（step 50和step 100等）
            if len(reconstructed_images) >= 2:
                # 按step排序重构图
                reconstructed_images.sort(key=lambda x: int(x.split('_step_')[-1].split('.png')[0]))

                # 打开原图和两张重构图
                original_image = Image.open(original_image_path)
                reconstructed_image_1 = Image.open(reconstructed_images[0])
                reconstructed_image_2 = Image.open(reconstructed_images[1])

                # 确保图片大小一致，选择最大的一张图片的尺寸作为组合图的尺寸
                max_width = max(original_image.width, reconstructed_image_1.width, reconstructed_image_2.width)
                max_height = max(original_image.height, reconstructed_image_1.height, reconstructed_image_2.height)

                # 调整图像为正方形（最大宽高）
                original_image = original_image.resize((max_width, max_height))
                reconstructed_image_1 = reconstructed_image_1.resize((max_width, max_height))
                reconstructed_image_2 = reconstructed_image_2.resize((max_width, max_height))

                # 创建一个新的图像，三张图并排
                combined_image = Image.new('RGB', (max_width * 3, max_height))

                # 将原图和重构图放入新的组合图中
                combined_image.paste(original_image, (0, 0))
                combined_image.paste(reconstructed_image_1, (max_width, 0))
                combined_image.paste(reconstructed_image_2, (max_width * 2, 0))

                # 保存组合图
                output_filename = f"{base_filename}.jpg"
                output_image_path = os.path.join(output_dir, output_filename)
                combined_image.save(output_image_path)

                print(f"组合图已保存: {output_image_path}")
                
                reconstructed_images_3 = Image.open(reconstructed_images[2])
                reconstructed_images_4 = Image.open(reconstructed_images[3])
                max_width = max(original_image.width, reconstructed_images_3.width, reconstructed_images_4.width)
                max_height = max(original_image.height, reconstructed_images_3.height, reconstructed_images_4.height)
                original_image = original_image.resize((max_width, max_height))
                reconstructed_images_3 = reconstructed_images_3.resize((max_width, max_height))
                reconstructed_images_4 = reconstructed_images_4.resize((max_width, max_height))
                combined_image = Image.new('RGB', (max_width * 3, max_height))
                combined_image.paste(original_image, (0, 0))
                combined_image.paste(reconstructed_images_3, (max_width, 0))
                combined_image.paste(reconstructed_images_4, (max_width * 2, 0))
                output_filename = f"{base_filename}_2.jpg"
                output_image_path = os.path.join(output_dir, output_filename)
                combined_image.save(output_image_path)
                print(f"组合图已保存: {output_image_path}")
                
    except Exception as e:
        print(f"处理 {filename} 时出错: {str(e)}")

print("处理完成！")