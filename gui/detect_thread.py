import numpy as np
import time
from PySide6.QtCore import QThread, Signal
from gui.predictor import UAVDETRPredictor

class DetectThread(QThread):
    # 发送推断结果
    # (rgb_img, num_targets, class_counts, time_info, total_time_ms)
    finished_signal = Signal(np.ndarray, int, dict, dict, float)
    
    # 错误信号
    error_signal = Signal(str)

    def __init__(self, weights_path, source_path, conf=0.25, iou=0.45, line_width=1, show_labels=True):
        super().__init__()
        self.weights_path = weights_path
        self.source_path = source_path
        self.conf = conf
        self.iou = iou
        self.line_width = line_width
        self.show_labels = show_labels
        
    def run(self):
        try:
            start_time = time.time()
            # 初始化预测器并在此线程中加载模型
            predictor = UAVDETRPredictor(self.weights_path)
            
            # 进行推断
            rgb_img, num_targets, class_counts, speed_info = predictor.predict(
                source=self.source_path,
                conf=self.conf,
                iou=self.iou,
                line_width=self.line_width,
                show_labels=self.show_labels
            )
            
            total_time_ms = (time.time() - start_time) * 1000
            
            # 发送结果数据到主线程
            if rgb_img is not None:
                self.finished_signal.emit(rgb_img, num_targets, class_counts, speed_info, total_time_ms)
            else:
                self.error_signal.emit("未检测到有效结果")
                
        except Exception as e:
            self.error_signal.emit(f"检测过程中发生错误: {str(e)}")
