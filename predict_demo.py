import sys
import os
import cv2

# 1. 【防坑必备】强制优先使用当前目录下的魔改代码
sys.path.insert(0, os.getcwd())
from ultralytics import RTDETR

if __name__ == '__main__':
    # 2. 指定你的模型权重路径
    weights_path = 'runs/train/r50/weights/best.pt'

    # 3. 指定你要检测的图片路径 (可以是一张图，也可以是一个包含多张图的文件夹)
    # 建议从 VisDrone 的 test 文件夹里挑一张小目标密集的图
    source_path = 'image_test'

    print(f"正在加载模型: {weights_path}")
    model = RTDETR(weights_path)

    print(f"开始检测图片: {source_path}")
    # 4. 执行预测
    results = model.predict(
        source=source_path,
        conf=0.25,  # 置信度阈值：低于 25% 概率的框会被过滤掉
        iou=0.45,  # NMS 阈值：用于过滤重叠框 (RT-DETR 默认其实不用NMS，但这算是个保险参数)
        save=True,  # 必须为 True，这样才会把画好框的图保存下来
        line_width=1,  # 画框的线条粗细（无人机图片目标小，建议设为 1 或 2，太粗会挡住物体）
        show_labels=True,  # 是否在框上显示类别名字
        show_conf=False,  # 是否在框上显示概率值（如果图里目标多，建议设为 False，不然画面全是字，很乱）
        project = 'runs/detect',  # 主文件夹
        name = 'predict',  # 子文件夹
        exist_ok = True  # 如果 predict 文件夹已存在，图片直接放进去，而不是新建 predict2、predict3
    )

    print("\n✅ 检测完成！请去 runs/detect/predict 文件夹下查看生成的图片。")