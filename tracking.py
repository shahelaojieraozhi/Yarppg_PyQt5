# corners = np.array([[281, 238], [325, 297], [283, 330], [248, 325]], dtype=np.float32).reshape(-1, 1, 2)
# H = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1], np.float32).reshape(3, 3)
# new = cv2.perspectiveTransform(corners, H)
# corners = np.array([281, 238, 1], dtype=np.float32)
# vec = np.matmul(H, corners)
# print(vec[0] / vec[2], vec[1] / vec[2])
# print(new[0][0])

# # # https://blog.csdn.net/weixin_45335726/article/details/122531876
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img = cv2.imread('./1.jpg')
# rows, cols, ch = img.shape
#
# '''
# pts1	原图像三个点的坐标
# pts2	原图像三个点在变换后相应的坐标
# '''
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv2.getAffineTransform(pts1, pts2)
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('output')
# plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 06:43:25 2017
@author: dc
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# 定义可视化图像函数
def look_img(img):
    '''opencv 读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# 读入图像
filename = './1.jpg'
img = cv2.imread(filename)
img2 = img
# 转化为灰度float32类型进行处理
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)
# 得到角点坐标向量
goodfeatures_corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
goodfeatures_corners = np.int0(goodfeatures_corners)

# 注意学习这种遍历的方法（写法）
for i in goodfeatures_corners:
    # 注意到i 是以列表为元素的列表，所以需要flatten或者ravel一下。
    x, y = i.flatten()
    cv2.circle(img2, (x, y), 15, [0, 255, ], -1)

look_img(img2)
