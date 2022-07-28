# fl = open(r'E:\data_set\My_dataset/Qt5_income.txt', encoding='utf-8')
# info = fl.read()
#
# # 薪资20000 以上 和 以下 的人员名单
# salary_above_20k = ''
# salary_below_20k = ''
# for line in info.splitlines():  # '薛蟠     4560 25'
#     if not line.strip():
#         continue
#     parts = line.split(' ')     # ['薛蟠', '', '', '', '', '4560', '25']
#     # 去掉列表中的空字符串内容
#     parts = [p for p in parts if p]  # ['薛蟠', '4560', '25']
#     name, salary, age = parts
#     if int(salary) >= 20000:
#         salary_above_20k += name + '\n'
#     else:
#         salary_below_20k += name + '\n'
#
# print(salary_above_20k)
# print(salary_below_20k)

# import random
#
# r_symbol = random.choice(['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star'])
# r_color = random.choice(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'd', 'l', 's'])
# print(r_symbol)


# import sys
#
# a = sys.argv[1]
# print(a)

# import sys
# from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QMainWindow,QHBoxLayout
# from PyQt5.QtCore import Qt, pyqtSignal
#
#
# class CMainWindow(QMainWindow):
#     signalTest = pyqtSignal()
#     signalTest1 = pyqtSignal(str)
#     signalTest2 = pyqtSignal(float, float)
#
#     def __init__(self):
#         super().__init__()
#         # 确认PushButton设置
#         btn = QPushButton("无参信号")
#         btn.clicked.connect(self.buttonClicked)
#         btn1 = QPushButton("单参信号")
#         btn1.clicked.connect(self.buttonClicked1)
#         btn2 = QPushButton('双参信号')
#         btn2.clicked.connect(self.buttonClicked2)
#         hBox = QHBoxLayout()    # 水平布局
#         hBox.addStretch(1)  # addStretch()函数用于在控件按钮间增加伸缩量
#         hBox.addWidget(btn)     # 往hBox里加三个按钮
#         hBox.addWidget(btn1)
#         hBox.addWidget(btn2)
#         widget = QWidget()      #
#         self.setCentralWidget(widget)
#         widget.setLayout(hBox)
#         self.signalTest.connect(self.signalNone)
#         self.signalTest1.connect(self.signalOne)
#         self.signalTest2.connect(self.signalTwo)
#         self.setWindowTitle('pysignal的使用')
#         self.show()
#
#     def signalNone(self):
#         print("无参信号，传来的信息")
#
#     def signalOne(self, arg1):
#         print("单参信号，传来的信息:", arg1)
#
#     def signalTwo(self, arg1, arg2):
#         print("双参信号，传来的信息:", arg1, arg2)
#
#     def mousePressEvent(self, event):
#         self.signalTest2.emit(event.pos().x(), event.pos().y())
#         # QWidget.pos()  获得窗口左上角的坐标
#         # 点击三个键的空白处输出：双参信号，传来的信息: 63.0 31.0
#
#     def buttonClicked(self):
#         self.signalTest.emit()
#         # 返回[]
#
#     def buttonClicked1(self):
#         self.signalTest1.emit("我是单参信号传来的")
#
#     def buttonClicked2(self):
#         self.signalTest2.emit(0, 0)
#
#     def keyPressEvent(self, e):
#         if e.key() == Qt.Key_Escape:
#             self.close()
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     MainWindow = CMainWindow()
#     sys.exit(app.exec_())

# import pyqtgraph.examples
#
# # 官方实例
# pyqtgraph.examples.run()


