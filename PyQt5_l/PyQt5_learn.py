from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox


class Stats():
    def __init__(self):
        self.window = QMainWindow()  # 主窗口
        self.window.resize(500, 400)
        self.window.move(300, 300)  # 出现在整个显示器的位置
        self.window.setWindowTitle('薪资统计')

        self.textEdit = QPlainTextEdit(self.window)  # text框
        self.textEdit.setPlaceholderText("请输入薪资表")  # 提示的字符——灰色字体一输入就消失了
        self.textEdit.move(10, 25)  # 文本框里在主窗口的位置(标题栏不算)
        self.textEdit.resize(300, 350)

        self.button = QPushButton('统计', self.window)
        self.button.move(380, 80)

        self.button.clicked.connect(self.handleCalc)  # 点击button会链接对应事件

    def handleCalc(self):
        info = self.textEdit.toPlainText()

        # 薪资20000 以上 和 以下 的人员名单
        salary_above_20k = ''
        salary_below_20k = ''
        for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')
            # 去掉列表中的空字符串内容
            parts = [p for p in parts if p]
            name, salary, age = parts
            if int(salary) >= 20000:
                salary_above_20k += name + '\n'
            else:
                salary_below_20k += name + '\n'

        QMessageBox.about(self.window,
                          '统计结果',
                          f'''薪资20000 以上的有：\n{salary_above_20k}
                    \n薪资20000 以下的有：\n{salary_below_20k}'''
                          )


app = QApplication([])  # 提供了整个图形界面程序的底层管理功能
stats = Stats()
stats.window.show()
app.exec_()  # 等待用户进行操作，只要不人为关掉，就一直在.  是个死循环

'''
薛蟠     4560 25
薛蝌     4460 25
薛宝钗   35776 23
薛宝琴   14346 18
王夫人   43360 45
王熙凤   24460 25
王子腾   55660 45
王仁     15034 65
尤二姐   5324 24
贾芹     5663 25
贾兰     13443 35
贾芸     4522 25
尤三姐   5905 22
贾珍     54603 35
'''
