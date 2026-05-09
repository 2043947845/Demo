STYLE_SHEET = """
QMainWindow {
    background-color: #f0f2f5;
}

QWidget {
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 18px;
    color: #303133;
}

/* 侧边栏和面板背景色 */
#ControlPanel, #ImagePanel {
    background-color: transparent;
}

/* 分组框设计 */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #ebeef5;
    border-radius: 10px;
    margin-top: 20px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 15px;
    top: 5px;
    color: #409eff;
    font-size: 20px;
}

/* 按钮设计 */
QPushButton {
    background-color: #409eff;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #66b1ff;
}

QPushButton:pressed {
    background-color: #3a8ee6;
}

QPushButton:disabled {
    background-color: #a0cfff;
    color: #ffffff;
}

/* 特定按钮样式 */
QPushButton#BtnDetect {
    background-color: #67c23a;
    font-size: 22px;
    border-radius: 8px;
}
QPushButton#BtnDetect:hover {
    background-color: #85ce61;
}
QPushButton#BtnDetect:pressed {
    background-color: #5daf34;
}

/* 输入框和下拉框等 */
QComboBox, QDoubleSpinBox, QTextEdit {
    border: 1px solid #dcdfe6;
    border-radius: 4px;
    padding: 5px;
    background-color: #ffffff;
    selection-background-color: #409eff;
}

QComboBox:hover, QDoubleSpinBox:hover, QTextEdit:hover {
    border: 1px solid #c0c4cc;
}

QComboBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 1px solid #409eff;
}

/* 下拉框展开按钮 */
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 0px;
}

/* 滚动条 */
QScrollBar:vertical {
    border: none;
    background: #f2f3f5;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #c0c4cc;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #909399;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* 图片显示区域 */
QLabel#ImageDisplay {
    background-color: #fafafa;
    border: 2px dashed #dcdfe6;
    border-radius: 8px;
    color: #909399;
}

QLabel#ImageDisplay:hover {
    border-color: #409eff;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
}
"""
