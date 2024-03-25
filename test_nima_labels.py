import json
import os
import tqdm

json_dir = "D:\\Dataset\\AVA\\nima_labels\\ava_labels_train.json"
images_dir = "D:\\Dataset\\AVA\\images"
new_json_dir = "D:\\Dataset\\AVA\\nima_labels\\ava_labels_train_filtered.json"

with open(json_dir, "r") as f:
    data = json.load(f)

    # 创建一个新的列表来保存存在的图像的信息
    filtered_data = []

    with tqdm.tqdm(total=len(data)) as pbar:
        for item in data:
            img_id = item['image_id']
            img_path = os.path.join(images_dir, str(img_id) + ".jpg")
            if os.path.exists(img_path):
                # 如果图像存在，则添加到filtered_data中
                filtered_data.append(item)
            else:
                # 如果图像不存在，可以在这里打印或处理
                print(f"Missing image: {img_path}")
            pbar.update(1)

# 保存新的json文件
with open(new_json_dir, "w") as f:
    json.dump(filtered_data, f)
