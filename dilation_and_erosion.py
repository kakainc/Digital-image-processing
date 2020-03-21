# --->
# Created by liumeiyu on 2020/3/16.
# '_'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from graphy import Graph

'''图像的腐蚀膨胀开闭运算'''


class D_E(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)
        self.img_b = cv2.threshold(self.gray_img, 127, 255, cv2.THRESH_BINARY)[1]

    def dilation(self, iters):
        kerl = np.ones((3, 3), dtype=np.uint8)
        img_d = cv2.dilate(self.img_b, kerl, iterations=iters)

        plt.subplot(121)
        plt.imshow(self.img_b, 'gray')
        plt.subplot(122)
        plt.imshow(img_d, 'gray')
        plt.show()

    def erosion(self, iters):
        kerl = np.ones((3, 3), dtype=np.uint8)
        img_e = cv2.erode(self.img_b, kerl, iterations=iters)

        plt.subplot(121)
        plt.imshow(self.img_b, 'gray')
        plt.subplot(122)
        plt.imshow(img_e, 'gray')
        plt.show()

    def morph(self):
        kerl = np.ones((3, 3), dtype=np.uint8)
        img_o = cv2.morphologyEx(self.img_b, cv2.MORPH_OPEN, kerl)  # 开运算
        img_c = cv2.morphologyEx(self.img_b, cv2.MORPH_CLOSE, kerl)  # 闭运算
        img_g = cv2.morphologyEx(self.img_b, cv2.MORPH_GRADIENT, kerl)  # 梯度运算(img) = 膨胀(img)- 腐蚀(img) 取轮廓
        img_top = cv2.morphologyEx(self.img_b, cv2.MORPH_TOPHAT, kerl)  # 顶帽运算(img) = 原始图像(img) - 开运算(img) 提取毛刺
        img_black = cv2.morphologyEx(self.img_b, cv2.MORPH_BLACKHAT, kerl)  # 黑帽运算(img) = 闭运算图像(img) - 原始图像(img) 提取空洞
        img_ = [self.img_b, img_o, img_c, img_g, img_top, img_black]
        for i in range(len(img_)):
            plt.subplot(231 + i)
            plt.imshow(img_[i], 'gray')
        plt.show()

    '''建造3d灰度图'''

    def morph_3d(self):
        fig = plt.figure(figsize=(24, 18))
        # ax = fig.gca(projection='3d')
        # ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig)

        x = np.arange(0, self.img_w, 1)
        y = np.arange(0, self.img_h, 1)
        x, y = np.meshgrid(x, y)  # 坐标.点.矩阵
        z = self.gray_img

        f_img = ax.plot_surface(x, y, z, cmap=plt.cm.get_cmap('coolwarm'))

        ax.set_zlim(-10, 255)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.set_xlabel('x', size=15)
        ax.set_ylabel('y', size=15)
        ax.set_title('surface_plot', weight='bold', size=20)

        fig.colorbar(f_img, shrink=0.6)

        plt.show()
