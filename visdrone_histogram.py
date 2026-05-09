import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 配置部分 ----------------------
ROOT_DIR = "dataset"  # 需修改为你的实际路径
SUBSETS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test"
]
CATEGORY_MAP = {
    1: "Pedestrian", 2: "People", 3: "Bicycle", 4: "Car", 5: "Van",
    6: "Truck", 7: "Tricycle", 8: "Awning-Tricycle", 9: "Bus", 10: "Motor"
}
# -------------------------------------------------------

# 1. 统计类别数量（与原代码一致）
category_names = list(CATEGORY_MAP.values())
subset_counts = {subset: defaultdict(int) for subset in SUBSETS}

for subset in SUBSETS:
    ann_dir = os.path.join(ROOT_DIR, subset, "annotations")
    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".txt"):
            continue
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                cat_id = int(parts[5])
                if cat_id in CATEGORY_MAP:
                    subset_counts[subset][CATEGORY_MAP[cat_id]] += 1

# 2. 整理数据（与原代码一致）
for subset in SUBSETS:
    counts = subset_counts[subset]
    subset_counts[subset] = [counts.get(cat, 0) for cat in category_names]

# 3. 绘制分组直方图 + 添加数值标签
x = np.arange(len(category_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))  # 稍微增大画布，避免标签拥挤
rects1 = ax.bar(x - width, subset_counts[SUBSETS[0]], width, label="Train")
rects2 = ax.bar(x, subset_counts[SUBSETS[1]], width, label="Val")
rects3 = ax.bar(x + width, subset_counts[SUBSETS[2]], width, label="Test-Dev")

# ---------------------- 关键修改：添加数值标签 ----------------------
ax.bar_label(rects1, padding=2, fontsize=9)  # padding: 标签与柱子顶部的距离
ax.bar_label(rects2, padding=2, fontsize=9)
ax.bar_label(rects3, padding=2, fontsize=9)
# ---------------------------------------------------------------------

# 美化图表
ax.set_xlabel("Object Category", fontsize=12)
ax.set_ylabel("Number of Instances", fontsize=12)
ax.set_title("Instance Count per Category in VisDrone Dataset", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(category_names, rotation=45, ha="right")
ax.legend()
fig.tight_layout()  # 自动调整布局，防止标签被截断

plt.show()