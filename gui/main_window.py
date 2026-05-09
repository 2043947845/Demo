import os
import cv2
import numpy as np
import json
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSlider, 
    QDoubleSpinBox, QGroupBox, QTextEdit, QMessageBox,
    QComboBox, QSizePolicy, QCheckBox, QDialog, QScrollArea,
    QGraphicsDropShadowEffect, QGraphicsView, QGraphicsScene
)
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor

from gui.detect_thread import DetectThread

class ImageLabel(QLabel):
    clicked = Signal()
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("ImageDisplay")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self._pixmap = None

    def set_image(self, pixmap):
        self._pixmap = pixmap
        if pixmap:
            self.setText("")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap and not self._pixmap.isNull():
            painter = QPainter(self)
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)

class ZoomView(QGraphicsView):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.addPixmap(pixmap)
        self.setScene(self.scene)
        
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.zoom_factor = 1.15
        
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale(self.zoom_factor, self.zoom_factor)
            else:
                self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
            event.accept()
        else:
            super().wheelEvent(event)
            
    def showEvent(self, event):
        super().showEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV-DETR Drone Detection System")
        self.resize(1400, 850)
        
        self.weights_path = ""
        self.source_image_path = ""
        self.detect_thread = None
        self.history_file = "model_history.json"
        self.model_history = []
        
        self.init_ui()
        
    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 20))
        widget.setGraphicsEffect(shadow)
        
    def init_ui(self):
        # 核心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- 左侧：控制面板 ---
        control_panel = QWidget()
        control_panel.setObjectName("ControlPanel")
        control_panel.setFixedWidth(380)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(20)
        main_layout.addWidget(control_panel)
        
        # 模型权重设置
        model_group = QGroupBox("模型设置 (Model Settings)")
        self.add_shadow(model_group)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(12)
        
        self.combo_weights = QComboBox()
        self.combo_weights.currentIndexChanged.connect(self.on_combo_weights_changed)
        
        self.btn_load_weights = QPushButton("浏览并加载权重文件 (.pt)")
        self.btn_load_weights.clicked.connect(self.browse_weights)
        
        model_layout.addWidget(QLabel("选择近期使用的模型:"))
        model_layout.addWidget(self.combo_weights)
        model_layout.addWidget(self.btn_load_weights)
        control_layout.addWidget(model_group)
        
        # 图像设置
        image_group = QGroupBox("图像输入 (Image Input)")
        self.add_shadow(image_group)
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(12)
        self.btn_load_image = QPushButton("选择图片 (Image)")
        self.btn_load_image.clicked.connect(self.load_image)
        image_layout.addWidget(self.btn_load_image)
        control_layout.addWidget(image_group)
        
        # 参数设置
        params_group = QGroupBox("检测参数 (Parameters)")
        self.add_shadow(params_group)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(12)
        
        # Conf threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度 (Conf):"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        conf_layout.addWidget(self.spin_conf)
        params_layout.addLayout(conf_layout)
        
        # IoU threshold
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU 阈值:"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.45)
        iou_layout.addWidget(self.spin_iou)
        params_layout.addLayout(iou_layout)
        
        # Show labels
        opts_layout = QHBoxLayout()
        self.chk_show_labels = QCheckBox("显示检测框标签 (Show Labels)")
        self.chk_show_labels.setChecked(True)
        opts_layout.addWidget(self.chk_show_labels)
        params_layout.addLayout(opts_layout)
        
        control_layout.addWidget(params_group)
        
        # 控制按钮 (单张检测在前)
        self.btn_detect = QPushButton(">> 单张开始检测 (Single) <<")
        self.btn_detect.setObjectName("BtnDetect")
        self.btn_detect.setMinimumHeight(45)
        self.add_shadow(self.btn_detect)
        font = QFont()
        font.setBold(True)
        self.btn_detect.setFont(font)
        self.btn_detect.clicked.connect(self.start_detection)
        control_layout.addWidget(self.btn_detect)

        # 批量处理设置 (批量处理在后)
        batch_group = QGroupBox("批量处理与报表导出 (Batch Processing)")
        self.add_shadow(batch_group)
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setSpacing(12)
        
        self.btn_load_batch = QPushButton("1. 选择批量图片文件夹")
        self.btn_load_batch.clicked.connect(self.load_batch_folder)
        batch_layout.addWidget(self.btn_load_batch)
        
        self.lbl_batch_path = QLabel("未选择文件夹...")
        self.lbl_batch_path.setWordWrap(True)
        self.lbl_batch_path.setStyleSheet("color: #909399;")
        batch_layout.addWidget(self.lbl_batch_path)
        
        from PySide6.QtWidgets import QProgressBar
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        batch_layout.addWidget(self.batch_progress)
        
        self.btn_batch_detect = QPushButton("2. 开始批量并导出报表")
        self.btn_batch_detect.setObjectName("BtnDetect")
        self.btn_batch_detect.setMinimumHeight(45)
        self.btn_batch_detect.clicked.connect(self.start_batch_detection)
        self.btn_batch_detect.setEnabled(False) # 默认禁用
        batch_layout.addWidget(self.btn_batch_detect)
        
        control_layout.addWidget(batch_group)
        
        # 加一个弹簧把组件往上推，避免拉伸窗体时纵向过度拉宽间距
        control_layout.addStretch()
        
        # --- 中间：图像显示面板 ---
        image_panel = QWidget()
        image_panel.setObjectName("ImagePanel")
        image_panel_layout = QVBoxLayout(image_panel)
        image_panel_layout.setContentsMargins(5, 15, 5, 15)
        image_panel_layout.setSpacing(20)
        main_layout.addWidget(image_panel, stretch=1)
        
        # 图像并排显示改成上下分布
        display_layout = QVBoxLayout()
        
        # 原始图像
        orig_group = QGroupBox("原始图片 (Original Image) - 点击图片可放大观察")
        self.add_shadow(orig_group)
        orig_layout = QVBoxLayout(orig_group)
        orig_layout.setContentsMargins(10, 20, 10, 10)
        self.lbl_orig_img = ImageLabel("等待导入...")
        self.lbl_orig_img.clicked.connect(lambda: self.show_zoom_dialog(self.lbl_orig_img._pixmap, "原始图片预览"))
        orig_layout.addWidget(self.lbl_orig_img)
        display_layout.addWidget(orig_group)
        
        # 结果图像
        res_group = QGroupBox("检测结果 (Detection Result) - 点击图片可放大、下载")
        self.add_shadow(res_group)
        res_layout = QVBoxLayout(res_group)
        res_layout.setContentsMargins(10, 20, 10, 10)
        self.lbl_res_img = ImageLabel("等待检测...")
        self.lbl_res_img.clicked.connect(lambda: self.show_zoom_dialog(self.lbl_res_img._pixmap, "检测结果预览", is_result=True))
        res_layout.addWidget(self.lbl_res_img)
        display_layout.addWidget(res_group)
        
        image_panel_layout.addLayout(display_layout)
        
        # --- 最右侧：日志面板 ---
        log_panel = QWidget()
        log_panel.setObjectName("ControlPanel")
        log_panel.setFixedWidth(300)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(10, 15, 15, 15)
        log_layout.setSpacing(10)
        
        log_label = QLabel("检测日志 (Logs):")
        font_lg = QFont()
        font_lg.setBold(True)
        log_label.setFont(font_lg)
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_btn_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton("清除日志")
        self.btn_clear_log.clicked.connect(self.clear_logs)
        self.btn_export_log = QPushButton("导出日志")
        self.btn_export_log.clicked.connect(self.export_logs)
        log_btn_layout.addWidget(self.btn_clear_log)
        log_btn_layout.addWidget(self.btn_export_log)
        log_layout.addLayout(log_btn_layout)
        
        main_layout.addWidget(log_panel)
        
        self.log("系统初始化完成。请加载权重文件和输入图片。")
        self.load_model_history()

    def load_model_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.model_history = json.load(f)
            except Exception as e:
                self.log(f"加载缓存失败: {str(e)}")
                self.model_history = []
        
        self.combo_weights.blockSignals(True)
        self.combo_weights.clear()
        self.combo_weights.addItem("请选择缓存的模型权重...")
        for p in self.model_history:
            self.combo_weights.addItem(os.path.basename(p), p)
            
        if self.model_history:
            self.combo_weights.setCurrentIndex(1)
            self.weights_path = self.model_history[0]
            self.log(f"已自动加载最近使用的模型: {os.path.basename(self.weights_path)}")
            
        self.combo_weights.blockSignals(False)

    def save_model_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"保存缓存失败: {str(e)}")

    def add_to_history(self, file_path):
        if file_path in self.model_history:
            self.model_history.remove(file_path)
        self.model_history.insert(0, file_path)
        self.model_history = self.model_history[:10]  # 最多存10条
        self.save_model_history()
        self.load_model_history()
        
    def on_combo_weights_changed(self, index):
        if index > 0:
            file_path = self.combo_weights.itemData(index)
            if file_path and os.path.exists(file_path):
                self.weights_path = file_path
                self.log(f"已切换权重: {os.path.basename(file_path)}")
            else:
                self.log("选中的权重文件不存在！")

    def browse_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型权重", "runs/", "PyTorch Weights (*.pt);;All Files (*)"
        )
        if file_path:
            self.weights_path = file_path
            self.log(f"已选择权重: {file_path}")
            self.add_to_history(file_path)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择输入图片", "image_test/", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_path:
            self.source_image_path = file_path
            self.show_image_on_label(self.source_image_path, self.lbl_orig_img)
            self.lbl_res_img.set_image(None)
            self.lbl_res_img.setText("等待检测...")
            self.log(f"已选择图片: {file_path}")

    def show_image_on_label(self, img_path_or_array, label):
        if isinstance(img_path_or_array, str):
            pixmap = QPixmap(img_path_or_array)
        else:
            # numpy array (RGB)
            h, w, c = img_path_or_array.shape
            bytes_per_line = c * w
            qimage = QImage(img_path_or_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
        label.set_image(pixmap)

    def start_detection(self):
        if not self.weights_path:
            QMessageBox.warning(self, "警告", "请先加载模型权重！")
            return
            
        self.add_to_history(self.weights_path)
            
        if not self.source_image_path:
            QMessageBox.warning(self, "警告", "请先选择需要检测的图片！")
            return
            
        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("检测中...")
        self.log("开始检测，请稍候...")
        
        conf = self.spin_conf.value()
        iou = self.spin_iou.value()
        show_labels = self.chk_show_labels.isChecked()
        
        self.detect_thread = DetectThread(
            weights_path=self.weights_path,
            source_path=self.source_image_path,
            conf=conf,
            iou=iou,
            show_labels=show_labels
        )
        
        self.detect_thread.finished_signal.connect(self.on_detection_finished)
        self.detect_thread.error_signal.connect(self.on_detection_error)
        self.detect_thread.start()

    def on_detection_finished(self, rgb_img, num_targets, class_counts, speed_info, total_time_ms):
        self.btn_detect.setEnabled(True)
        if hasattr(self, 'batch_folder_path') and self.batch_folder_path:
            self.btn_batch_detect.setEnabled(True)
        self.btn_detect.setText(">> 单张开始检测 (Single) <<")
        
        # 显示结果图像
        self.show_image_on_label(rgb_img, self.lbl_res_img)
        
        # 打印日志
        self.log(f"检测成功! 发现 {num_targets} 个目标。")
        if class_counts:
            counts_str = " | ".join([f"{k}: {v}" for k, v in class_counts.items()])
            self.log(f"检测详情: {counts_str}")
            
        self.log(f"总耗时: {total_time_ms:.2f} ms")
        if speed_info:
            self.log(f"其中 -> 预处理:{speed_info.get('preprocess', 0):.1f}ms, 推理:{speed_info.get('inference', 0):.1f}ms, 后处理:{speed_info.get('postprocess', 0):.1f}ms")

    def on_detection_error(self, err_msg):
        self.btn_detect.setEnabled(True)
        if hasattr(self, 'batch_folder_path') and self.batch_folder_path:
            self.btn_batch_detect.setEnabled(True)
        self.btn_detect.setText(">> 单张开始检测 (Single) <<")
        self.log(f"错误: {err_msg}")
        QMessageBox.critical(self, "检测失败", err_msg)

    def load_batch_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含无人机图片的文件夹", "")
        if folder_path:
            self.batch_folder_path = folder_path
            self.lbl_batch_path.setText(f"已选: {folder_path}")
            self.btn_batch_detect.setEnabled(True)
            self.log(f"批量文件夹: {folder_path}")

    def start_batch_detection(self):
        if not hasattr(self, 'batch_folder_path') or not self.batch_folder_path:
            QMessageBox.warning(self, "警告", "请先选择需要处理的文件夹！")
            return
            
        if not self.weights_path:
            QMessageBox.warning(self, "警告", "请先加载模型权重！")
            return
            
        self.btn_batch_detect.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.batch_progress.setValue(0)
        self.log(f"开始对 {self.batch_folder_path} 进行扫描与报表生成...")
        
        # 导入刚才创建的线程
        from gui.batch_thread import BatchDetectThread
        self.batch_thread = BatchDetectThread(
            weights_path=self.weights_path,
            folder_path=self.batch_folder_path,
            conf=self.spin_conf.value(),
            iou=self.spin_iou.value()
        )
        self.batch_thread.progress_signal.connect(self.on_batch_progress)
        self.batch_thread.finished_signal.connect(self.on_batch_finished)
        self.batch_thread.error_signal.connect(self.on_batch_error)
        self.batch_thread.start()

    def on_batch_progress(self, current, total, filename):
        percentage = int(current / total * 100)
        self.batch_progress.setValue(percentage)
        self.log(f"[{current}/{total}] 处理完毕: {filename}")

    def on_batch_finished(self, csv_path, total_processed):
        self.btn_batch_detect.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.log(f"批量操作圆满完成！共成功处理并识别 {total_processed} 张图片。")
        self.log(f"数据汇总大屏报表已生成至: {csv_path}")
        QMessageBox.information(self, "批处理完成", f"共处理图片: {total_processed} 张。\n报表与框选结果已成功导出至:\n{csv_path}")

    def on_batch_error(self, err_msg):
        self.btn_batch_detect.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.log(f"批量执行错误中止: {err_msg}")
        QMessageBox.critical(self, "出错", err_msg)

    def log(self, text):
        self.log_text.append(text)
        self.log_text.ensureCursorVisible()

    def clear_logs(self):
        self.log_text.clear()
        self.log("日志已清除。")

    def export_logs(self):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"logs_{timestamp}.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出日志", default_path, "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "导出成功", f"日志已成功导出到:\n{file_path}")
                self.log(f"日志已导出至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出日志失败:\n{str(e)}")
                self.log(f"导出日志失败: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        
    def show_zoom_dialog(self, pixmap, title, is_result=False):
        if not pixmap or pixmap.isNull():
            return
            
        dlg = QDialog(self)
        dlg.setWindowTitle(title + " (按住Ctrl+鼠标滚轮缩放，鼠标拖动平移)")
        
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        
        view = ZoomView(pixmap)
        layout.addWidget(view)
        
        if is_result:
            bottom_layout = QHBoxLayout()
            bottom_layout.setContentsMargins(15, 10, 15, 15)
            
            btn_save = QPushButton("下载到本地 (Save to Local)")
            btn_save.setObjectName("BtnDetect")
            btn_save.setMinimumHeight(40)
            btn_save.clicked.connect(lambda: self.save_result_image(pixmap))
            
            bottom_layout.addStretch()
            bottom_layout.addWidget(btn_save)
            bottom_layout.addStretch()
            
            layout.addLayout(bottom_layout)
            
        dlg.showMaximized()
        dlg.exec()
        
    def save_result_image(self, pixmap):
        import datetime
        save_dir = "image_result"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = os.path.join(save_dir, f"result_{timestamp}.jpg")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存检测结果", default_path, "Images (*.jpg *.png *.jpeg *.bmp);;All Files (*)"
        )
        if file_path:
            pixmap.save(file_path)
            self.log(f"已保存检测结果至: {file_path}")
            QMessageBox.information(self, "保存成功", f"图片已保存至:\n{file_path}")
        
