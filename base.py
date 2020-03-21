# --->
# Created by liumeiyu on 2020/3/15.
# '_'

from graphy import Graph
import cv2
import matplotlib.pyplot as plt

'''图像的放缩镜像'''


class Base(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def change1(self):
        # c_img = self.img_t
        # c_img[1000:2500, 1000:1500] = [255, 255, 255]
        #
        # face = self.img_t[1000:2000, 1000:2000]

        b, g, r = cv2.split(self.img)
        img = cv2.merge([r, g, b])

        plt.imshow(img)
        plt.show()

    def change2(self):
        # r_img = cv2.resize(self.img_t, (int(self.img_h*0.6), int(self.img_w*0.4)))
        r_img = cv2.resize(self.img_t, None, fx=0.6, fy=0.3)

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(r_img)
        plt.show()

    def change3(self):
        f_img = cv2.flip(self.img_t, -1)  # 0,1,-1

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(f_img)
        plt.show()
