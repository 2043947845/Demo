import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ===================== 仅需修改这里的路径 =====================
ROOT_PATH = "dataset/UAVVaste"
# ============================================================

ALL_ANNOTATIONS = f"{ROOT_PATH}/annotations/annotations.json"
SPLIT_FILE = f"{ROOT_PATH}/annotations/train_val_test_distribution_file.json"

# 1. 读取全部标注和类别信息
with open(ALL_ANNOTATIONS, "r", encoding="utf-8") as f:
    anno_data = json.load(f)

# 建立关键映射：文件名 -> 图片ID (这是修复的核心！)
filename_to_id = {img["file_name"]: img["id"] for img in anno_data["images"]}
# 类别ID -> 类别名称映射
cat_map = {cat["id"]: cat["name"] for cat in anno_data["categories"]}
cat_names = list(cat_map.values())

# 2. 读取数据集划分，并把“文件名”转成“图片ID”
with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split_data = json.load(f)

# 转换：文件名 -> ID
train_img_ids = set([filename_to_id[fname] for fname in split_data["train"] if fname in filename_to_id])
val_img_ids = set([filename_to_id[fname] for fname in split_data["val"] if fname in filename_to_id])
test_img_ids = set([filename_to_id[fname] for fname in split_data["test"] if fname in filename_to_id])

# 3. 按子集统计每个类别的数量
train_cnt = defaultdict(int)
val_cnt = defaultdict(int)
test_cnt = defaultdict(int)

# 图片ID对应所属子集
img_to_subset = {}
for img_id in train_img_ids:
    img_to_subset[img_id] = "train"
for img_id in val_img_ids:
    img_to_subset[img_id] = "val"
for img_id in test_img_ids:
    img_to_subset[img_id] = "test"

# 遍历所有标注
for ann in anno_data["annotations"]:
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    cat_name = cat_map[cat_id]

    # 判断归属哪个集合
    subset = img_to_subset.get(img_id, None)
    if subset == "train":
        train_cnt[cat_name] += 1
    elif subset == "val":
        val_cnt[cat_name] += 1
    elif subset == "test":
        test_cnt[cat_name] += 1

# 对齐所有类别顺序
train_list = [train_cnt.get(c,0) for c in cat_names]
val_list = [val_cnt.get(c,0) for c in cat_names]
test_list = [test_cnt.get(c,0) for c in cat_names]

# ---------------- 绘制分组柱状图 ----------------
x = np.arange(len(cat_names))
width = 0.28
fig, ax = plt.subplots(figsize=(15,8))

# 画三组柱子
rect1 = ax.bar(x - width, train_list, width, label="Train Set")
rect2 = ax.bar(x, val_list, width, label="Val Set")
rect3 = ax.bar(x + width, test_list, width, label="Test Set")

# 柱子顶部添加具体数值
ax.bar_label(rect1, padding=3, fontsize=8)
ax.bar_label(rect2, padding=3, fontsize=8)
ax.bar_label(rect3, padding=3, fontsize=8)

# 图表美化
ax.set_xlabel("Category Name", fontsize=13)
ax.set_ylabel("Instance Count", fontsize=13)
ax.set_title("UAVVaste Dataset: Category Distribution (Train/Val/Test)", fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(cat_names, rotation=45, ha="right")
ax.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)
fig.tight_layout()

# 保存高清图+展示
plt.savefig("UAVVaste_category_hist.png", dpi=300, bbox_inches="tight")
plt.show()

# 额外：终端打印统计表格，方便核对
print("===== 类别统计结果 =====")
for i, name in enumerate(cat_names):
    print(f"{name:15} | Train:{train_list[i]:6} | Val:{val_list[i]:5} | Test:{test_list[i]:5}")