# --->
# Created by liumeiyu on 2020/3/18.
# '_'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from graphy import Graph

'''图像的灰度变换'''


class Cvt(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def change_color(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        img_g = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # 在plt下显示正常的灰色
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)  # HSV包含Hue（色调）、Saturation（饱和度）、Value（亮度）
        img_hls = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)  # HLS包含Hue(色相)、Luminance(亮度)、Saturation(饱和度)
        img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)  # Lab包含Luminance(亮度)和a、b两个颜色通道。
        img_ = [self.img_t, img_gray, img_g, img_hsv, img_hls, img_lab]
        for i in range(len(img_)):
            plt.subplot(231 + i)
            plt.imshow(img_[i])
        plt.show()

    # 人眼对绿色的敏感最高，对蓝色敏感最低，因此，RGB三分量进行加权平均可得到较合理的灰度图像。
    # gray = 0.299*R + 0.587*G + 0.144*B
    # a包括的颜色是从深绿色（低亮度值）到灰色（中亮度值）再到亮粉红色（高亮度值）；b是从亮蓝色（低亮度值）到灰色（中亮度值）再到黄色（高亮度值）

    # 图像灰度线性变换
    # f(c) = a*f(o) + b
    # a=1,  b=0    : 图像保持不变
    # a=1,  b！=0  : 图像灰度值上移或下移
    # a=-1, b=255  : 图像灰度值反转
    # a>1          : 图像对比度增强
    # 0<a<1        : 图像对比度减弱
    # a<0          : 图像亮暗互补
    # 对数变换 暗部提升
    # gamma变化 参数>1 拉伸亮部，压缩暗部
    #          参数<1 拉伸暗部，压缩亮部

    def change_gray(self):
        img_gc1 = self.gray_img * (-1)

        img_gc2 = np.log(self.gray_img + 1.0)
        lut = np.zeros(256, dtype=np.float32)
        for i in range(256):
            lut[i] = i ** 0.5
        img_gc3 = cv2.LUT(self.gray_img, lut)  # 映射

        plt.subplot(221)
        plt.imshow(self.gray_img, 'gray')
        plt.subplot(222)
        plt.imshow(img_gc1, 'gray')
        plt.subplot(223)
        plt.imshow(img_gc2, 'gray')
        plt.subplot(224)
        plt.imshow(img_gc3, 'gray')
        plt.show()
