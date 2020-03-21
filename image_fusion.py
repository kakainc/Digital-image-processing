# --->
# Created by liumeiyu on 2020/3/20.
# '_'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''图像的加法以及融合'''


class Fusion(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def img_fusion(self):
        img_np = np.add(self.img_t, self.img_t)  # 越界取模
        img_cv = cv2.add(self.img_t, self.img_t)  # 越界饱和（255）

        img_f = cv2.addWeighted(self.img, 0.4, self.img_t, 0.8, gamma=0)  # gamma亮度调节量

        plt.imshow(img_np)
        plt.show()
        plt.imshow(img_cv)
        plt.show()
        plt.subplot(121)
        plt.imshow(self.img)
        plt.subplot(122)
        plt.imshow(img_f)
        plt.show()
