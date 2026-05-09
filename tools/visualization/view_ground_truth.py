import cv2
import os
# 核心秘籍：直接导入 Ultralytics 官方的颜色分配器
from ultralytics.utils.plotting import colors


def draw_ground_truth_matching(img_path, label_path, save_path):
    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 图片读取失败，请检查路径: {img_path}")
        return
    h, w, _ = img.shape

    # 2. 读取同名的 YOLO 格式 txt 标签
    if not os.path.exists(label_path):
        print(f"❌ 找不到对应的标签文件: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"这张图里共有 {len(lines)} 个真实目标...")

    # 3. 遍历每一个框并绘制
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue

        class_id = int(data[0])

        # 【关键修改】：调用官方调色板
        # 传入 class_id，bgr=True 保证输出的是 OpenCV 需要的 蓝-绿-红 格式
        color = colors(class_id, bgr=True)

        # 读取归一化坐标并反算像素坐标
        x_c, y_c, box_w, box_h = map(float, data[1:5])
        center_x = int(x_c * w)
        center_y = int(y_c * h)
        width = int(box_w * w)
        height = int(box_h * h)

        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)

        # 画框。线宽设为 1，与你之前预测时的 line_width=1 保持一致
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # 4. 保存结果
    cv2.imwrite(save_path, img)
    print(f"✅ 完美同色版 GT 图已保存至: {save_path}")


if __name__ == '__main__':
    # ---------------------------------------------------------
    # 填入你要处理的图片名
    image_name = '0000129_02411_d_0000138.jpg'
    # ---------------------------------------------------------

    img_file = f'../../image_test/{image_name}'
    txt_file = f'../../dataset/visdrone_yolo/labels/val/{image_name.replace(".jpg", ".txt")}'
    save_file = f'../../runs/detect/predict/GT_Match_{image_name}'

    draw_ground_truth_matching(img_file, txt_file, save_file)