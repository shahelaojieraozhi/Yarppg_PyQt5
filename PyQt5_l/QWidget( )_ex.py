# # -*- coding: utf-8 -*-
#
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
# import sys
#
# app = QApplication(sys.argv)
# widget = QWidget()
# btn = QPushButton(widget)
# btn.setText("Button")
# btn.move(20, 20)
# widget.resize(300, 200)
# widget.move(250, 200)
# widget.setWindowTitle("PyQt坐标系统例子")
# widget.show()
# print("QWidget:")
# print("w.x() = %d " % widget.x())
# print("w.y() = %d " % widget.y())
# print("w.width() = %d " % widget.width())
# print("w.height = %d " % widget.height())
# print("QWidget.geometry")
# # x()，y()获得客户区左上角的坐标，width()、height()获得客户区的宽度和高度
# print("widget.geometry().x() = %d " % widget.geometry().x())
# print("widget.geometry().y() = %d " % widget.geometry().y())
# print("widget.geometry().width() = %d " % widget.geometry().width())
# print("widget.geometry().height() = %d " % widget.geometry().height())
# sys.exit(app.exec_())

# # -*- coding: utf-8 -*-
# import sys
# from PyQt5.QtWidgets import QApplication, QWidget
#
# app = QApplication(sys.argv)
# window = QWidget()
# window.resize(300, 200)
# window.move(250, 150)
# window.setWindowTitle("Hello PyQt5")
# window.show()
# sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# import sys
# from PyQt5.QtGui import QIcon
# from PyQt5.QtWidgets import QWidget, QApplication
#
#
# # 创建一个Icon的窗口，继承来自QWidget类
# class Icon(QWidget):
#     def __init__(self, parent=None):
#         super(Icon, self).__init__(parent)
#         self.initUI()
#
#     # 初始化窗口
#     def initUI(self):
#         self.setGeometry(300, 300, 250, 150)
#         self.setWindowTitle("程序图标")
#         self.setWindowIcon(QIcon('./a.ico'))
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     icon = Icon()
#     icon.show()
#     sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# import sys
# from PyQt5.QtWidgets import QWidget, QToolTip, QApplication
# from PyQt5.QtGui import QFont
#
#
# class winform(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUi()
#
#     def initUi(self):
#         QToolTip.setFont(QFont('SansSerif', 10))
#         self.setToolTip("这是一个<b>气泡提示</b>")
#         self.setGeometry(200, 300, 400, 400)
#         self.setWindowTitle("气泡提示demo")
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     win = winform()
#     win.show()
#     sys.exit(app.exec_())
