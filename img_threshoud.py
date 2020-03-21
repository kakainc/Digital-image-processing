# --->
# Created by liumeiyu on 2020/3/16.
# '_'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from graphy import Graph

'''图像二值化'''


class Binary(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def img_biny_np(self, thrd):
        img_b = self.copy_img()
        for i in range(self.img_h):
            for j in range(self.img_w):
                if self.gray_img[i][j] > thrd:
                    img_b[i][j] = 255
                else:
                    img_b[i][j] = 0
        return img_b

    def biny(self, thrd):
        r, img_b1 = cv2.threshold(self.gray_img, thrd, 255, cv2.THRESH_BINARY)  # r阈值
        r, img_b2 = cv2.threshold(self.gray_img, thrd, 255, cv2.THRESH_BINARY_INV)
        r, img_b3 = cv2.threshold(self.gray_img, thrd, 255, cv2.THRESH_TRUNC)  # 大于thrd为thrd， 反之不变
        r, img_b4 = cv2.threshold(self.gray_img, thrd, 255, cv2.THRESH_TOZERO)
        r, img_b5 = cv2.threshold(self.gray_img, thrd, 255, cv2.THRESH_TOZERO_INV)
        img_b6 = cv2.adaptiveThreshold(self.gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 501, 0)
        img_b7 = cv2.adaptiveThreshold(self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 0)
        r, img_b8 = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 自动计算双峰图的谷底阈值
        print(r)
        img_ = [self.gray_img, img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7, img_b8]
        for i in range(len(img_)):
            plt.subplot(331 + i)
            plt.imshow(img_[i], 'gray')
        plt.show()

    # 迭代选取最佳阈值

    def get_thrd(self, ts):
        total_gray = np.sum(self.gray_img)
        hist_g = cv2.calcHist(self.gray_img, [0], None, [256], [0, 255])
        hist_g = np.reshape(hist_g, (1, -1))[0]
        T1 = total_gray // self.gray_img.size
        T2 = 0
        while abs(T2 - T1) > ts:
            s1, n1, s2, n2 = 0, 0, 0, 0
            for i, j in enumerate(hist_g):
                if i < T1:
                    s1 += j * i
                    n1 += j
                else:
                    s2 += j * i
                    n2 += j
            T2 = (s1 / n1 + s2 / n2) / 2
            T1, T2 = T2, T1
        return T1
