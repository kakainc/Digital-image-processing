# --->
# Created by liumeiyu on 2020/3/20.
# '_'

import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''向上采样和向下采样'''


class Sampling(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    # 对图像进行高斯核卷积，并删除原图中所有的偶数行和列，图像缩小为1/4
    def down_sampling(self):
        img_d1 = cv2.pyrDown(self.img_t)
        img_d2 = cv2.pyrDown(img_d1)
        img_d3 = cv2.pyrDown(img_d2)

        plt.imshow(self.img_t)
        plt.show()
        plt.imshow(img_d3)
        plt.show()

    # 先扩大为原图像的4倍，新增的行和列均用0来填充，并使用4倍的高斯核与放大后的图像进行卷积运算
    def up_sampling(self):
        img_u = cv2.pyrUp(self.img_t)

        plt.imshow(self.img_t)
        plt.show()
        plt.imshow(img_u)
        plt.show()
