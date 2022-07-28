# 导入了一个自定义模块后，就可以直接使用里面的变量或者函数等
# import my_module
# print(my_module.name)
# # 张三
#
# my_module.ha()
# # hello

# from my_module import *
# # *表示通配符，代表了自定义模块中所有的变量和方法等
# print(name)
# print(age)
# ha()

# # 第三种方式：
# from my_module import name, age
# print(name)
# print(age)

# class Student:
#     def eat(self):
#         print(self)
#
#
# stu = Student()
# stu.eat()   # 此时self表示的是stu这个对象
# # <__main__.Student object at 0x00000225E2519AC8>
#
# stu1 = Student()
# stu1.eat()  # 此时的self表示的是stu1这个对象
# # <__main__.Student object at 0x0000023F270D9188>
# #
# # # self 表示的是当前对象
# '''
# 1.不需要手动调用，会在合适的时机自动触发
# 2.这些方法都是使用__ 开始和使用__结束
# 3。方法名都是系统规定好的
# '''
#
#
# class Animal():
#     '''
#     类属性（不推荐这样写）
#     name = "老虎"
#     sex = "雄性"
#     '''
#
#     # 构造函数：当创建了对象之后，给对象赋值属性的时候自动触发
#     def __init__(self, name, sex):
#         self.name = name
#         self.sex = sex
#         print("我的触发时机是：当你给对象的属性赋值的时候自动触发：", name)
#         print(self.name, self.sex)
#
#     # 析构函数：当对象被销毁的时候，自动触发
#     def __del__(self):
#         print("我的触发时机是：当对象被销毁的时候触发")
#
#
# tigger = Animal("老虎", "雄性")
# print("hello")
# # hello
# # 我的触发时机是：当对象被销毁的时候触发

# class Fish():
#     # 类属性
#     name = "鱼类"
#
#     # 构造函数 对象属性：
#     def __init__(self, weight):
#         self.weight = weight
#
#     def swin(self):
#         print("游泳的方法")
#
#
# # 创建对象
# jinyu = Fish(234)
# liyu = Fish(875)

# # 访问类属性     （对象和类都可以访问类属性）
# print(Fish.name)    # 鱼类
# print(jinyu.name)   # 鱼类
# print(liyu.name)    # 鱼类

# 访问对象的属性   (对象可以访问对象的属性，但是类不能访问对象的属性)
# print(jinyu.weight)     # 234
# print(liyu.weight)      # 875
# print(Fish.weight)      # 报错

"""
1.类属性可以使用类名访问（推荐）对象也可以访问类属性（不推荐）
2.对象的属性可以使用对象访问（推荐)，类不可以访问对象的属性
"""

# # 修改类属性
# Fish.name = "鲨鱼"
# print(Fish.name)    # 鲨鱼
#
# jinyu.name = "金鱼"
# print(jinyu.name)   # 金鱼
#
# print(liyu.name)    # 鲨鱼

# # 修改对象的属性
# jinyu.weight = "1斤"
# print(jinyu.weight)     # 1斤
# Fish.weight = "500克"
# print(Fish.weight)      # 500克

# # 最基本的继承
#
# class Person():
#     def say(self):
#         print("说话的方法")
#
#
# class Boy(Person):  # 在子类中继承父类只需要定义子类时，参数列表中写上父类的名称即可
#
#     def eat(self):
#         print("吃饭的方法")
#
#
# # 创建一个子类的对象
# xiaoming = Boy()
# xiaoming.eat()  # 对象自己调用子类的方法
# xiaoming.say()  # 对象调用父类的方法

# #  有构造函数的继承
# class Animal():
#     def __init__(self, name, sex):
#         self.name = name
#         self.sex = sex
#
#     def run(self):
#         print("我是跑的方法")
#
#
# class Dog(Animal):
#     def __init__(self, name, sex, weight):  # 先在子类的构造函数中继承父类的属性，然后再重构
#
#         # # 2.隐式继承父类的构造函数
#         super(Dog, self).__init__(name, sex)
#
#         # 定义子类自己的属性
#         self.weight = weight
#
#     def wang(self):
#         print("汪汪！")
#
#
# taidi = Dog("泰迪", "公", 5000)
# taidi.run()
# print(taidi.name)
# print(taidi.sex)
#
# # 我是跑的方法
# # 泰迪
# # 公

# # 定义一个类
# class Person(object):
#     def __init__(self, name):
#         self.name = name
#
#     def run(self):
#         print("跑步的方法")
#
#
# '''
# 注意：
#     1.object是所有类的父类，如果一个类没有显示指明他的父类，则默认为object.(可以省略不写)
#     2.python中的面向对象可以实现多继承，
# '''
#
# person = Person("张三")
# person.run()
# print(person.name)

# 多继承语法：
'''
class 子类类名(父类1，父类2，父类3.....)
    属性
    方法
'''


# # 演示_定义一个父亲类
# class Father():
#     def __init__(self, surname):
#         self.surname = surname
#
#     def make_money(self):
#         print("钱难挣！")
#
#
# # 定义一个母亲类
# class Monther():
#     def __init__(self, height):
#         self.height = height
#
#     def eat(self):
#         print("一言不合就干饭！")
#
#
# # 定义子类
# class Son(Father, Monther):  # 子类继承多个父类时，只需要把父类的名称写在参数列表中即可，中间用逗号隔开
#     def __init__(self, surname, height, weight):
#         # 继承父类的构造函数
#         Father.__init__(self, surname)
#         Monther.__init__(self, height)
#         # 定义子类的属性
#         self.weight = weight
#
#     def play(self):
#         print("打王者")
#
#
# son = Son("xiaoming", 178, 130)
# print(son.surname)
# print(son.height)
# son.make_money()
# son.play()
# # xiaoming


# test()
#
#
# def test():
#     print("你好吗？")
#     print("我很好")
#
#
# test()


#
#
# def demo():
#     test()
#     print("我很好")
#
#
# demo()


# def outer(fn):
#     def inner():
#         fn()
#         print("我很好")
#
#     return inner
#
#
# # 装饰器的简写方式: @ + 装饰器的名称
#
# # 原函数:
# @outer  # 等价于 test = outer(test)
# def test():
#     print("你好吗？")
#
#
# test()
#
# '''
# 注意：
# 1.在使用装饰器简写方式的时候，原函数必须在装饰器函数的下面
# 2.outer函数就是装饰器函数， @outer ===》 test = outer(test)
# '''

# # 定义一个装饰器函数，做下面的数学运算的函数添加解释说明
# def outer(fn):
#     def inner(*args):
#         print("数学运算的结果是：", end=" ")
#         fn(*args)
#
#     return inner
#
#
# @outer
# def add(a, b):
#     print(a + b)
#
#
# @outer
# def subtract(c, d, e):
#     print(c - d - e)
#
#
# add(12, 34)
# subtract(87, 56, 6)

# def outer1(fn):
#     def inner1():
#         fn()
#         print("给我听吧！老妹！")
#
#     return inner1
#
#
# def outer2(fn):
#     def inner2():
#         fn()
#         print("滚呐.......")
#
#     return inner2
#
#
# # 换位置也对应变化——离原函数越近就越先输出
# @outer2
# @outer1
# def say():
#     print("唱歌个谁听.....")
#
#
# say()

# def test():
#     print("hello world")
#
#
# demo = test     # 当把函数名赋值给一个变量的时候，那这个变量就实现了和函数一样的功能
# demo()

# def add(x, y):
#     print(x + y)
#
#
# def substract(a, b):
#     print(a - b)
#
#
# def multiply(c, d):
#     print(c * d)
#
#
# def divide(e, f):
#     print(e / f)
#
#
# # add(12, 34)
# # substract(76, 32)
#
#
# # 需求：封装一个万能函数，传入两个参数，直接实现加减乘除的操作
# def demo(x, y, func):
#     func(x, y)
#
#
# # 加法运算
# demo(12, 23, add)
#
# demo(12, 23, substract)

# # 最简单的闭包实现
# def outer():
#     def inner():
#         print("我是闭包函数！")
#
#     return inner  # 注意：这里返回的是函数体，不是函数的调用
#
#
# fn = outer()  # fn ====> inner
# fn()  # 相当于调用了inner函数


# # 闭包的小练习：
# def outer1(x):
#     y = 11
#
#     # 闭包特点：内部函数可以使用外部变量
#     def inner1():
#         print(x + y)
#
#     return inner1
#
#
# func1 = outer1(7)
# func1()
#
# # 闭包函数主要用于装饰器函数的实现

# if 5 > 4:
#     a = 11
# print(a)

# for i in range(1, 11):
#     b = 77
# print(b)


# num1 = 67
#
#
# def fn1():
#     # 如想在函数的内部直接修改函数外部的变量，需要使用global关键字，将函数内部变量变更为全局变量
#     global num1
#     num1 = 88
#     print(num1)
#
#
# fn1()   # 88
# print(num1)     # 88


# class Animal():
#     # 类属性
#     name = "动物类"
#
#     def __init__(self, name):
#         self.name = name
#
#     @classmethod
#     def run(cls):
#         print("我是类方法")
#         print(cls)
#         print(cls == Animal)  # cls表示的是当前类
#         print(cls.name)
#
#     @staticmethod
#     def eat():
#         print("我是静态方法")
#
#
# Animal.eat()    # 通过类名调用静态方法
# # 我是静态方法

# class Girl():
#     def __init__(self, name, sex, age):
#         self.name = name
#         self.sex = sex
#         # 比如女孩的年龄是私密的，不能在类的外部轻易地访问，那可以设置为私有属性
#         # 私有属性：在属性的前面加 __
#         self.__age = age
#
#     def say(self):
#         print("说话的方法")
#
#     # 私有方法：__方法名
#     def __kiss(self):
#         print("一吻定终生")
#
#     def love(self, relationship):
#         if relationship == "情侣":
#             self.__kiss()
#         else:
#             print("不能随便kiss，小心中毒")
#
#     # 在类的内部定义一个方法，可以访问私有属性
#     def sayAge(self, boyFriend):
#         if boyFriend == "小明":
#             print(f"{self.name}偷偷地告诉{boyFriend}说：我今年18岁了")
#         else:
#             print("女孩子的年龄是秘密")
#
#
# hong = Girl("小红", "女", 18)
# print(hong.name)  # 小红
# print(hong.sex)  # 女
# # print(hong.age)  # 报错 ——   在类的外部不能直接访问对象的私有属性.
#
# hong.say()  # 说话的方法
# hong.sayAge("小亮")  # 女孩子的年龄是秘密
# hong.sayAge("小明")  # 小红偷偷地告诉小明说：我今年18岁了
#
# # hong.__kiss()   # 私有方法外部不能直接访问
# hong.love("情侣")     # 一吻定终生

# # 父类
# class Animal():
#     def eat(self):
#         print("吃饭的方法")
#
#
# # 子类
# class Fish(Animal):
#     def eat(self):
#         print("大鱼吃小鱼，小鱼吃虾米")
#
#
# class Dog(Animal):
#     def eat(self):
#         print("狼行千里吃肉，狗星万里吃粑粑")
#
#
# class Cat(Animal):
#     def eat(self):
#         print("猫爱吃鱼")
#
#
# # 严格意义的多态的体现
# class Person():
#     def feed(self, animal):
#         animal.eat()
#
#
# # 最简单的多态的体现
# fish = Fish()
# dog = Dog()
# cat = Cat()
#
# # fish.eat()  # 大鱼吃小鱼，小鱼吃虾米
# # dog.eat()  # 狼行千里吃肉，狗星万里吃粑粑
# # cat.eat()  # 猫爱吃鱼
#
# Person().feed(dog)
# Person().feed(cat)
# # 狼行千里吃肉，狗星万里吃粑粑
# # 猫爱吃鱼

# 单例设计模式

# class Person():
#     def __init__(self, name):
#         self.name = name
#
#     # 创建一个类属性，接受创建的对象
#     instance = None
#
#     @classmethod
#     def __new__(cls, *args, **kwargs):
#         print("__new__")
#         #   如果类属性的instance == None 表示，该类未创建对象
#         if cls.instance == None:
#             cls.instance = super().__new__(cls)
#         return cls.instance
#
#
# p1 = Person("成灭那个")
# p2 = Person("zhangsan")
# p3 = Person("李四")
#
# print(p1 == p2 == p3)


