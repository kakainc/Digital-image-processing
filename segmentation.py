# --->
# Created by liumeiyu on 2020/3/19.
# '_'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''基于K_means的图像分割'''


class Segmentation(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def k_means(self):
        # img_o = np.reshape(self.gray_img, (self.img_h * self.img_w, 1))
        img_o = np.reshape(self.img_t, (-1, 3))
        img_o = np.float32(img_o)

        # 算法终止条件，即达到最大迭代次数或所需精度(type,max_iter,epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # 初始中心的选择
        flag = cv2.KMEANS_RANDOM_CENTERS

        # compactness：每个点到他们对应的中心的距离的平方和
        # labels: 标签数组(图像123)
        # centers: 聚类中心的数组
        compactness, labels, centers = cv2.kmeans(img_o, K=3, bestLabels=None, criteria=criteria, attempts=10,
                                                  flags=flag)
        print(compactness)
        print(centers)
        print(labels.shape)  # 在3维上求距离得到1维结果

        # img_k = labels.reshape(self.img_h, self.img_w)

        centers = np.uint8(centers)
        data = centers[labels.flatten()]  # 将1维结果在3维上投影得到3维结果
        img_k = data.reshape(self.img_t.shape)

        plt.subplot(111)
        plt.imshow(img_k)
        plt.title("segmentation")
        plt.show()
