import cv2
import numpy as np
import os
import math

# ==========================================
# 参数配置区域
# ==========================================
CROP_LEFT = 500
SRC_CENTER_ORIG = (969, 489)
SRC_RADIUS = 100
# 放大倍数调大后，建议把目标左下角往下放一点（比如750），防止大圆超出上边界
TGT_BOTTOM_LEFT_ORIG = (1200, 650)

MAGNIFICATION = 3  # 放大倍数
GUIDE_COLOR = (0, 0, 255)
GUIDE_THICKNESS = 2  # 圆圈边界的粗细
LINE_THICKNESS = 2  # <--- 新增：切线连线的粗细 (设为2，比之前的1更醒目)


# ==========================================
# 核心功能函数定义
# ==========================================

def create_circular_inset(img_path, save_name):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 读取图片失败: {img_path}")
        return None

    # 1. 裁剪左侧
    img = img[:, CROP_LEFT:]
    h, w, _ = img.shape
    output_img = img.copy()

    # 2. 坐标动态转换
    src_cx = SRC_CENTER_ORIG[0] - CROP_LEFT
    src_cy = SRC_CENTER_ORIG[1]
    src_center = (src_cx, src_cy)

    tgt_bl_x = TGT_BOTTOM_LEFT_ORIG[0] - CROP_LEFT
    tgt_bl_y = TGT_BOTTOM_LEFT_ORIG[1]

    tgt_radius = int(SRC_RADIUS * MAGNIFICATION)
    patch_size_mag = tgt_radius * 2
    tgt_center = (tgt_bl_x + tgt_radius, tgt_bl_y - tgt_radius)

    tgt_x1 = tgt_bl_x
    tgt_x2 = tgt_bl_x + patch_size_mag
    tgt_y1 = tgt_bl_y - patch_size_mag
    tgt_y2 = tgt_bl_y

    if src_cx - SRC_RADIUS < 0 or tgt_x2 > w or tgt_y1 < 0:
        print(f"❌ 坐标越界！裁剪后图片宽为 {w}，请检查目标放置位置。")
        return None

    # 3. 提取、放大并裁剪出圆形 Patch
    patch = img[src_cy - SRC_RADIUS: src_cy + SRC_RADIUS,
            src_cx - SRC_RADIUS: src_cx + SRC_RADIUS].copy()
    patch_mag = cv2.resize(patch, (patch_size_mag, patch_size_mag), interpolation=cv2.INTER_LINEAR)

    mask = np.zeros((patch_size_mag, patch_size_mag), dtype=np.uint8)
    cv2.circle(mask, (tgt_radius, tgt_radius), tgt_radius, 255, -1)
    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # --------------------------------------------------
    # 第四步：计算并绘制外切线 (完美“放大镜”物理透视效果)
    # --------------------------------------------------
    dx = tgt_center[0] - src_center[0]
    dy = tgt_center[1] - src_center[1]
    d = math.hypot(dx, dy)  # 计算圆心距

    # 确保两个圆不完全重合才能画切线
    if d > abs(tgt_radius - SRC_RADIUS):
        theta = math.atan2(dy, dx)
        phi = math.acos((tgt_radius - SRC_RADIUS) / d)

        # 计算源圆上的两个切点
        pt1_src = (int(src_center[0] + SRC_RADIUS * math.cos(theta + phi)),
                   int(src_center[1] + SRC_RADIUS * math.sin(theta + phi)))
        pt2_src = (int(src_center[0] + SRC_RADIUS * math.cos(theta - phi)),
                   int(src_center[1] + SRC_RADIUS * math.sin(theta - phi)))

        # 计算目标圆(大圆)上的两个切点
        pt1_tgt = (int(tgt_center[0] + tgt_radius * math.cos(theta + phi)),
                   int(tgt_center[1] + tgt_radius * math.sin(theta + phi)))
        pt2_tgt = (int(tgt_center[0] + tgt_radius * math.cos(theta - phi)),
                   int(tgt_center[1] + tgt_radius * math.sin(theta - phi)))

        # 绘制切线连线 (使用 LINE_THICKNESS = 2 加粗)
        cv2.line(output_img, pt1_src, pt1_tgt, GUIDE_COLOR, LINE_THICKNESS)
        cv2.line(output_img, pt2_src, pt2_tgt, GUIDE_COLOR, LINE_THICKNESS)

    # --------------------------------------------------
    # 第五步：合成与描边
    # --------------------------------------------------
    # 1. 画源位置的红圈
    cv2.circle(output_img, src_center, SRC_RADIUS, GUIDE_COLOR, GUIDE_THICKNESS)

    # 2. 将放大的圆形图像印到目标位置上 (这步会盖住切线在圆内的部分，显得更立体)
    target_roi = output_img[tgt_y1:tgt_y2, tgt_x1:tgt_x2]
    np.copyto(target_roi, patch_mag, where=mask_3c == 255)

    # 3. 画放大后位置的红圈边界
    cv2.circle(output_img, tgt_center, tgt_radius, GUIDE_COLOR, GUIDE_THICKNESS)

    print(f"✓ 已生成圆形切线放大图: {save_name}")
    return output_img


# ==========================================
# 脚本主程序逻辑
# ==========================================
if __name__ == '__main__':
    gt_path = '../../runs/detect/predict/GT_Match_0000129_02411_d_0000138.jpg'
    pred_path = '../../runs/detect/predict/test03.jpg'
    final_output_name = '../../runs/detect/predict/Final_Tangent_Comparison.jpg'

    processed_gt = create_circular_inset(gt_path, 'temp_gt.jpg')
    processed_pred = create_circular_inset(pred_path, 'temp_pred.jpg')

    if processed_gt is not None and processed_pred is not None:
        h, _, _ = processed_gt.shape
        divider = np.ones((h, 10, 3), dtype=np.uint8) * 255

        final_img = cv2.hconcat([processed_gt, divider, processed_pred])
        cv2.imwrite(final_output_name, final_img)
        print(f"\n✅ 完美！终极切线横向拼接大图已保存至: {final_output_name}")