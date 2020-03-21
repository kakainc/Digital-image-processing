# --->
# Created by liumeiyu on 2020/3/17.
# '_'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''图像的透视旋转'''


class Warp(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def warpA(self):
        # M1 = cv2.getRotationMatrix2D((self.img_h // 2, self.img_w // 2), 45, 1)  # 旋转
        # M2 = np.float32([[1, 0, -1000], [0, 1, -1000]])                          # 平移 左右上下

        pos1 = np.float32([[100, 100], [100, 500], [500, 100]])  # 定下三个点
        pos2 = np.float32([[120, 120], [100, 400], [400, 90]])
        M3 = cv2.getAffineTransform(pos1, pos2)  # 图像仿射变换

        img_warp_t = cv2.warpAffine(self.img_t, M3, (self.img_h, self.img_w))

        pos1 = np.float32([[100, 100], [100, 500], [500, 100], [500, 500]])  # 定下四个点
        pos2 = np.float32([[120, 120], [100, 400], [400, 90], [400, 400]])
        M4 = cv2.getPerspectiveTransform(pos1, pos2)  # 图像透视变换

        img_warp_p = cv2.warpPerspective(self.img_t, M4, (self.img_h, self.img_w))

        plt.subplot(121)
        plt.imshow(img_warp_t)
        plt.subplot(122)
        plt.imshow(img_warp_p)
        plt.show()
