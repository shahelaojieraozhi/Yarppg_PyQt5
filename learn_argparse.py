import argparse

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
# # parser.parse_args(['--sum', '7', '-1', '42'])

# -*- coding: utf-8 -*-

# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("square", help="display a square of a given number", type=int)
# args = parser.parse_args()
# print(args.square ** 2)

# import argparse
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--weights', nargs='+', type=str, default='yo', help='modle')
# parser.add_argument('--weights', action='store_ture', help='modle')
#
# # '--weights' 参数名称，后面是对应的属性
# # nargs='+' —— 当前命令行消耗的数目：有’+‘——至少一个参数，如果是’？‘就是0个或者一个 如果是’*‘就是0个或者所有
# # type=str ——传进来的参数什么类型
# # default='yo' —— 如果没有传入值，使用这个默认值
# # help='modle' —— 写帮助信息的
# # action='strod' —— 当参数出现在命令窗口时，他的动作类型： store_ture默认值是False  store_False默认值是True

# # cylinder 圆柱体
# import math
#
#
# def cylinder_volume(radius, height):
#     vol = (math.pi) * (radius ** 2) * (height)
#     return vol
#
#
# if __name__ == '__main__':
#     print(cylinder_volume(2, 4))
#
#
import math
import argparse

parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
parser.add_argument('radius', type=int, help='Radius of Cylinder')
parser.add_argument('height', type=int, help='Height of Cylinder')
args = parser.parse_args()


def cylinder_volume(radius, height):
    vol = (math.pi) * (radius ** 2) * (height)
    return vol


if __name__ == '__main__':
    print(cylinder_volume(args.radius, args.height))
