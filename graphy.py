# --->
# Created by liumeiyu on 2020/3/10.
# '_'

import cv2
import numpy as np

'''建立图像基类，包括BGR RGB GRAY 图像，图像高宽，以及图像等大小的白板'''
'''np.uint8 np.float32'''


class Graph:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)  # cv2
        self.img_t = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # plt
        self.img_h, self.img_w, _ = self.img.shape
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def copy_img(self):
        copy_img = np.array([[255] * self.img_w for _ in range(self.img_h)], dtype=np.uint8)
        return copy_img
