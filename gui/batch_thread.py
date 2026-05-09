import os
import time
import csv
import datetime
import numpy as np
from PySide6.QtCore import QThread, Signal
from gui.predictor import UAVDETRPredictor

class BatchDetectThread(QThread):
    # 发送进度信息 (当前处理的索引, 总数, 当前图片名)
    progress_signal = Signal(int, int, str)
    
    # 全部完成信号，返回生成的 CSV 文件路径和处理成功的总数
    finished_signal = Signal(str, int)
    
    # 错误信号
    error_signal = Signal(str)

    def __init__(self, weights_path, folder_path, conf=0.25, iou=0.45, line_width=1):
        super().__init__()
        self.weights_path = weights_path
        self.folder_path = folder_path
        self.conf = conf
        self.iou = iou
        self.line_width = line_width
        self._is_running = True
        
    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # 筛选出图像文件
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            image_files = []
            if not os.path.exists(self.folder_path):
                self.error_signal.emit("目录不存在！")
                return
                
            for f in os.listdir(self.folder_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_exts:
                    image_files.append(f)
                    
            if not image_files:
                self.error_signal.emit("该目录下未找到任何支持的图片文件！")
                return
                
            total_images = len(image_files)
            
            # 载入预测器模型 (在线程内部实例化防止冲突发生)
            predictor = UAVDETRPredictor(self.weights_path)
            
            # CSV 存放设置
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(self.folder_path, f"batch_report_{timestamp}.csv")
            
            # 检测结果图像的临时保存处
            save_dir = os.path.join(self.folder_path, f"batch_results_{timestamp}")
            
            records = []
            
            for idx, img_name in enumerate(image_files):
                if not self._is_running:
                    break
                    
                self.progress_signal.emit(idx + 1, total_images, img_name)
                
                img_path = os.path.join(self.folder_path, img_name)
                
                if idx == 0:
                    os.makedirs(save_dir, exist_ok=True)
                
                # 开始调用单张推理逻辑
                rgb_img, num_targets, class_counts, speed_info = predictor.predict(
                    source=img_path,
                    conf=self.conf,
                    iou=self.iou,
                    line_width=self.line_width,
                    show_labels=True
                )
                
                # 统计和归档预测结果
                details = " | ".join([f"{k}:{v}" for k, v in class_counts.items()]) if class_counts else "无目标"
                
                # 写出带有边界框的结果图片
                if rgb_img is not None:
                    import cv2
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir, img_name), bgr_img)
                
                records.append({
                    "Image Name": img_name,
                    "Total Targets Found": num_targets,
                    "Details (Class:Count)": details,
                    "Preprocess Time (ms)": round(speed_info.get("preprocess", 0), 2) if speed_info else 0,
                    "Inference Time (ms)": round(speed_info.get("inference", 0), 2) if speed_info else 0,
                    "Postprocess Time (ms)": round(speed_info.get("postprocess", 0), 2) if speed_info else 0
                })
                
            # 当整个序列运行结束后写入最终的数据报表
            if records:
                with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "Image Name", "Total Targets Found", "Details (Class:Count)", 
                        "Preprocess Time (ms)", "Inference Time (ms)", "Postprocess Time (ms)"
                    ])
                    writer.writeheader()
                    writer.writerows(records)
                    
            if self._is_running:
                self.finished_signal.emit(csv_path, len(records))
            
        except Exception as e:
            self.error_signal.emit(f"批量处理过程中遇到致命错误: {str(e)}")
