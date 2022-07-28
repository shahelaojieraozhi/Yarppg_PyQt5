# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# import sys
#
#
# class MyTable(QTableWidget):
#     def __init__(self, parent=None):
#         super(MyTable, self).__init__(parent)
#         self.setWindowTitle("me")
#         self.setShowGrid(False)  # 设置显示格子线
#         # self.setStyleSheet("QTableWidget{background-color: white;border:20px solid #014F84}")
#         self.setStyleSheet("QTableWidget{background-color: black;border:20px solid #014F84}"
#                            "QTableWidget::item{border:1px solid #014F84}")
#
#         self.resize(1000, 600)
#         self.setColumnCount(5)  # 设置5列
#         self.setRowCount(2)     # 2行
#         self.setColumnWidth(0, 220)     #
#         self.setColumnWidth(1, 220)
#         self.setColumnWidth(2, 220)
#         self.setColumnWidth(4, 300)
#
#         self.setRowHeight(0, 100)
#         # 设置第一行高度为100px,第一列宽度为200px
#         self.table()
#
#     def table(self):
#         # self指的是MyTable这个类
#         # self.setStyleSheet("Box{border:5px}")
#         Item00 = QTableWidgetItem("2018/11/09 10:45\nXXX欢迎使用X号工作台")
#         textFont = QFont("song", 14, QFont.Bold)    # 宋体14号 QFont.Bold
#         Item00.setFont(textFont)
#         self.setItem(0, 0, Item00)
#
#         # self.resizeColumnsToContents()
#         # self.resizeRowsToContents()#行和列的大小设置为与内容相匹配
#         Item01 = QTableWidgetItem("九亭1号仓")
#         textFont = QFont("song", 19, QFont.Bold)
#         Item01.setFont(textFont)
#         self.setItem(0, 1, Item01)
#         Item02 = QTableWidgetItem("美菜 土豆 3KG")
#         textFont = QFont("song", 19, QFont.Bold)
#         Item02.setFont(textFont)
#         self.setItem(0, 2, Item02)
#
#         button = QPushButton()
#         Item03 = QTableWidgetItem("退出")  # 在这里面需要加一个按钮，按钮为红色，按钮文字为退出
#         textFont = QFont("song", 13, QFont.Bold)
#         button.setFont(textFont)
#         button.setObjectName("button")
#         button.setStyleSheet("#button{background-color: red}")
#         Item03.setFont(textFont)
#         self.setItem(0, 3, Item03)
#
#         self.verticalHeader().setVisible(False)  # 影藏列表头
#         self.horizontalHeader().setVisible(False)  # 隐藏行表头
#
#         # 下面设置表格的边框颜色
#         self.item(0, 0).setForeground(QBrush(QColor(255, 255, 255)))
#         self.item(0, 0).setForeground(QBrush(QColor(255, 255, 255)))  # 设置字体的颜色，还需要设置字体的大小
#         self.item(0, 1).setForeground(QBrush(QColor(255, 255, 255)))
#         self.item(0, 2).setForeground(QBrush(QColor(255, 255, 255)))
#         self.item(0, 3).setForeground(QBrush(QColor(255, 255, 255)))
#
#         # self.item(0,4).setForeground(QBrush(QColor(255, 255, 255)))
#
#
# app = QApplication(sys.argv)
# mytable = MyTable()
# mytable.show()
# app.exec()
