import json
import os

#  VisDrone 验证集真实标签路径
gt_path = r"../../dataset\VisDrone_COCO\annotations\instances_test_vis.json"
#  predictions.json 路径
pred_path = r"../../runs\test\test_uav_detr_r50\predictions.json"
# 保存的新文件名
fixed_pred_path = r"../../runs\test\test_uav_detr_r50\predictions_fixed.json"
# ==============================================================

print("正在读取文件，请稍候...")

# 1. 读取真实标签，建立“文件名 -> 数字 ID”的映射字典
with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

name2id = {}
for img in gt_data['images']:
    # 去掉 .jpg 后缀，提取纯名字，例如 "0000001_02999_d_0000005"
    base_name = os.path.splitext(img['file_name'])[0]
    name2id[base_name] = img['id']

# 2. 读取预测结果，把长名字替换成数字 ID
with open(pred_path, 'r', encoding='utf-8') as f:
    preds = json.load(f)

valid_preds = []
for p in preds:
    old_id = p['image_id']
    # 尝试把旧的字符串 ID 转换为数字 ID
    # 注意：有时候预测结果里的 old_id 可能本来就是数字字符串，所以先转成 string 比较稳妥
    old_id_str = str(old_id).replace('.jpg', '')

    if old_id_str in name2id:
        # 修复一：替换真实的数字 image_id
        p['image_id'] = name2id[old_id]

        # 修复二：将预测的 category_id 加 1，对齐 COCO 官方标签！
        # (YOLO输出是0-9, VisDrone COCO通常是1-10)
        p['category_id'] = p['category_id'] + 1

        valid_preds.append(p)

    # 3. 保存修复后的文件
with open(fixed_pred_path, 'w', encoding='utf-8') as f:
    json.dump(valid_preds, f)

print(f"修复完成！成功转换了 {len(valid_preds)} 个预测框。")
print(f"请使用新文件 {fixed_pred_path} 重新运行 get_COCO_metrice.py。")