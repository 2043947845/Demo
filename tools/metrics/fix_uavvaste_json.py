import json
import os

# ================== 请在这里填入你的文件路径 ==================
# 1. 你刚刚生成的干净的 UAVVaste 验证集标签
gt_path = r"../../dataset\UAVVaste\annotations\uavvaste_val.json"

# 2. 你的模型在 UAVVaste 验证集上跑出来的预测文件 (⚠️请务必修改这个路径！)
pred_path = r"../../runs\val\val_r50_uavvaste\predictions.json"

# 3. 修复后保存的新文件名 (运行后会生成在当前目录)
fixed_pred_path = r"../../runs\val\val_r50_uavvaste\predictions_fixed.json"
# ==============================================================

print("正在读取文件，请稍候...")

with open(gt_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

# 1. 动态获取真实的类别 ID (自动适应数据集是 0 还是 1)
real_category_id = gt_data['categories'][0]['id']
print(f"检测到真实标签中的类别 ID 为: {real_category_id}")

# 2. 建立图片 ID 映射字典
name2id = {}
for img in gt_data['images']:
    # 提取核心文件名，去掉后缀
    base_name = os.path.splitext(str(img.get('file_name', str(img.get('id')))))[0]
    name2id[base_name] = img['id']

# 3. 读取并修复预测结果
with open(pred_path, 'r', encoding='utf-8') as f:
    preds = json.load(f)

valid_preds = []
for p in preds:
    old_id = str(p['image_id']).replace('.jpg', '')

    # 如果预测的图片ID能在验证集里找到
    if old_id in name2id:
        # 修复一：替换真实的数字 image_id
        p['image_id'] = name2id[old_id]
        # 修复二：强制统一类别 ID (因为 UAVVaste 只有一个类，直接对齐标签里的真实ID)
        p['category_id'] = real_category_id

        valid_preds.append(p)

with open(fixed_pred_path, 'w', encoding='utf-8') as f:
    json.dump(valid_preds, f)

print(f"修复完成！成功转换了 {len(valid_preds)} 个预测框。")
print(f"请使用新文件 {fixed_pred_path} 重新运行 get_COCO_metrice.py。")