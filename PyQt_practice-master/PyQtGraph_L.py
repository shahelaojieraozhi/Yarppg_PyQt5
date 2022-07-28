# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg
#
# # 创建 绘制窗口类 PlotWindow 对象，内置一个绘图控件类 PlotWidget 对象
# pw = pg.plot()
#
# # 设置图表标题、颜色、字体大小
# pw.setTitle("气温趋势", color='008080', size='12pt')
#
# # 背景色改为白色
# pw.setBackground('w')
#
# # 显示表格线
# pw.showGrid(x=True, y=True)
#
# # 设置上下左右的label
# # 第一个参数 只能是 'left', 'bottom', 'right', or 'top'
# pw.setLabel("left", "气温(摄氏度)")
# pw.setLabel("bottom", "时间")
#
# # 设置Y轴 刻度 范围
# pw.setYRange(min=-10,  # 最小值
#              max=50)  # 最大值
#
# # 创建 PlotDataItem ，缺省是曲线图
# curve = pw.plot(pen=pg.mkPen('b'))  # 线条颜色
#
# hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
#
# curve.setData(hour,  # x坐标
#               temperature  # y坐标
#               )
#
# # 清除原来的plot内容
# pw.clear()
#
# # 创建 PlotDataItem ，缺省是曲线图
# curve = pw.plot(pen=pg.mkPen('b'))  # 线条颜色
# hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# temperature = [130, 132, 134, 132, 133, 131, 129, 132, 135, 145]
#
# curve.setData(hour,  # x坐标
#               temperature  # y坐标
#               )
#
# QtGui.QApplication.instance().exec_()

from PyQt5 import QtWidgets
import pyqtgraph as pg


class MainWindow(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('pyqtgraph作图示例')

        # 创建 PlotWidget 对象
        self.pw = pg.PlotWidget()

        # 设置图表标题
        self.pw.setTitle("气温趋势", color='008080', size='12pt')

        # 设置上下左右的label
        self.pw.setLabel("left", "气温(摄氏度)")
        self.pw.setLabel("bottom", "时间")
        # 背景色改为白色
        self.pw.setBackground('w')

        hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]

        # hour 和 temperature 分别是 : x, y 轴上的值
        self.pw.plot(hour,
                     temperature,
                     pen=pg.mkPen('b')  # 线条颜色
                     )

        # 创建其他Qt控件
        okButton = QtWidgets.QPushButton("OK")
        lineEdit = QtWidgets.QLineEdit('点击信息')
        # 水平layout里面放 edit 和 button
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(lineEdit)
        hbox.addWidget(okButton)

        # 垂直layout里面放 pyqtgraph图表控件 和 前面的水平layout
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.pw)
        vbox.addLayout(hbox)

        # 设置全局layout
        self.setLayout(vbox)


if __name__ == '__main__':
    app = QtWidgets.QApplication()
    main = MainWindow()
    main.show()
    app.exec_()
