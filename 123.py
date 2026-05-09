import json

# 改成你的路径
ROOT_PATH = "dataset/UAVVaste"
ALL_ANNOTATIONS = f"{ROOT_PATH}/annotations/annotations.json"
SPLIT_FILE = f"{ROOT_PATH}/annotations/train_val_test_distribution_file.json"

# 1. 看看划分文件长啥样
print("--- 1. train_val_test_distribution_file.json 的内容 ---")
with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split_data = json.load(f)
print("Keys:", list(split_data.keys()))
print("Train集前3个元素:", split_data.get("train", [])[:3])
print("Val集前3个元素:", split_data.get("val", [])[:3])

# 2. 看看标注文件长啥样
print("\n--- 2. annotations.json 的内容 ---")
with open(ALL_ANNOTATIONS, "r", encoding="utf-8") as f:
    anno_data = json.load(f)
print("Keys:", list(anno_data.keys()))
print("\n--- 前2张图片信息 ---")
for img in anno_data.get("images", [])[:2]:
    print(img)
print("\n--- 前2个标注信息 ---")
for ann in anno_data.get("annotations", [])[:2]:
    print(ann)