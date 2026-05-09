import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.style import STYLE_SHEET

def main():
    app = QApplication(sys.argv)
    
    # 设置应用风格
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
