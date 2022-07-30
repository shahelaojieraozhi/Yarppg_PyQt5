from PyQt5.QtWidgets import QApplication, QWidget, QLabel


class Window(QWidget):
    def __init__(self):
        super().__init__()  # 继承了Qwidget的属性

        self.setWindowTitle("社会你治哥")
        self.resize(400, 400)
        self.setup_ui()

        # 将所有的添加子控件操作和子控件的配对操作全部放到这

    def setup_ui(self):
        label = QLabel(self)
        label.setText("xxx")
        label.move(200, 200)


# 为什么要用 if __name__ == '__main__': 因为现在这个文件是个封装好的类了
# if __name__ == '__main__':  下面的代码是调用上面写的类 用来测试的。如果不这样写，
# 把这个类当作其他的文件的模块时会自动运行下面的代码。
if __name__ == '__main__':      # 判断当前程序是被直接执行还是被导入执行
    import sys
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())

# 这里还讲了一下活动模板设置


