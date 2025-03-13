from PIL import Image
import os
from pathlib import Path

# 定义路径
path1 = "/data2/changshuochen/model/generated_images_new"   # 右侧图片路径
path2 = "/data2/changshuochen/model/style30k/images"    # 左侧图片路径
output_dir = "/data2/changshuochen/model/combined_images_new"  # 输出目录

# 支持的图片格式
supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 创建输出目录
Path(output_dir).mkdir(parents=True, exist_ok=True)

# 获取path1中的所有图片文件
for filename in os.listdir(path1):
    # 检查文件扩展名
    if filename.lower().endswith(supported_ext):
        path1_file = os.path.join(path1, filename)
        path2_file = os.path.join(path2, filename)
        
        # 确保path2中存在对应文件
        if not os.path.exists(path2_file):
            print(f"跳过 {filename}，路径2中不存在对应文件")
            continue

        try:
            # 打开两张图片并转换为RGB模式
            with Image.open(path2_file) as left_img, Image.open(path1_file) as right_img:
                left_img = left_img.convert('RGB')
                right_img = right_img.convert('RGB')

                # 调整右侧图片高度与左侧一致
                left_height = left_img.height
                right_aspect = right_img.width / right_img.height
                new_right_width = int(left_height * right_aspect)
                right_resized = right_img.resize((new_right_width, left_height))

                # 创建拼接后的新图片
                total_width = left_img.width + right_resized.width
                combined = Image.new('RGB', (total_width, left_height))
                combined.paste(left_img, (0, 0))
                combined.paste(right_resized, (left_img.width, 0))

                # 保存结果
                output_path = os.path.join(output_dir, filename)
                combined.save(output_path)
                print(f"已保存: {output_path}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")