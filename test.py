# import pyqtgraph.examples
#
# # 官方实例
# pyqtgraph.examples.run()

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        print("xxxx")


app = QApplication(sys.argv)

window = Window()

window.show()

sys.exit(app.exec_())
