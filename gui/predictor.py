import os
import sys
import cv2
import numpy as np

# 强制优先使用当前目录下的魔改代码
sys.path.insert(0, os.getcwd())
from ultralytics import RTDETR

class UAVDETRPredictor:
    def __init__(self, weights_path):
        """
        初始化预测器并加载模型
        """
        self.weights_path = weights_path
        self.model = RTDETR(weights_path)
        
    def predict(self, source, conf=0.25, iou=0.45, line_width=1, show_labels=True):
        """
        执行检测并返回带有绘制框的图像 (RGB格式 NumPy 数组)
        :param source: 图像路径
        :param conf: 置信度阈值
        :param iou: NMS 阈值
        :param line_width: 线条宽度
        :param show_labels: 是否显示标签和置信度
        :return: (检测后的RGB图数组, 目标数量, 各类别数量字典, 耗时信息字典)
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=False,  # 不在后端自动保存，由前端 GUI 接管展示
        )
        
        if not results:
            return None, 0, {}, {}
            
        result = results[0]
        # result.plot() 返回的是 BGR 格式的 numpy 数组图像，带有检测框
        bgr_img = result.plot(line_width=line_width, labels=show_labels, conf=show_labels)
        # 转换为 RGB 格式以供 PySide 展示
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # 目标数量
        num_targets = len(result.boxes)
        
        # 统计各类别数量
        class_counts = {}
        if result.boxes and result.names:
            for box in result.boxes:
                # 获取类别 ID，并将其转换为整数
                cls_id = int(box.cls[0].item())
                # 获取该类别 ID 对应的名称
                cls_name = result.names.get(cls_id, f"Class {cls_id}")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # 耗时信息提取 (预处理、推断、后处理毫秒数)
        speed_info = result.speed 
        
        return rgb_img, num_targets, class_counts, speed_info
