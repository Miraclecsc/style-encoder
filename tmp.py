import json
import random

def replace_captions(input_json_path, output_json_path):
    # 准备一些简单描述性的短语
    descriptions = [
        "A dog sitting on the table",
        "A cat resting by the window",
        "A people playing in the park",
        "A living room with a table",
        "A church with a cross on top",
        "A street with cars and people",
        "A city with tall buildings",
        "A forest with tall trees",
        "A field with green grass",
        "A river with flowing water",
        "A beach with gentle waves",
        "A garden full of flowers",
        "A mountain trail",
        "A person reading under a tree",
        "A birds flying across the sky",
        "A child eating on the floor",
        "A small boat sailing on a lake",
        "A couple dancing in living room"
    ]
    
    # 读取原始 JSON 文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 对每个对象的 caption 进行随机替换
    for item in data:
        new_caption = random.choice(descriptions)
        item['caption'] = f"{new_caption}, in the style of sks."
    
    # 将修改后的数据写入新的 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 示例用法：假设原始文件为 'input.json'，输出文件为 'output.json'
    replace_captions('/data2/changshuochen/model/style30k/image_captions.json', '/data2/changshuochen/model/style30k/new.json')