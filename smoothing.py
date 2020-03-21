# --->
# Created by liumeiyu on 2020/3/10.
# '_'

from graphy import Graph
import matplotlib.pyplot as plt
import cv2

'''图像平滑除噪'''


class Smooth(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def done_linear(self, i, j):
        point = self.gray_img[i - 1][j - 1] / 9 + self.gray_img[i - 1][j] / 9 + self.gray_img[i - 1][j + 1] / 9 + \
                self.gray_img[i][j - 1] / 9 + self.gray_img[i][j] / 9 + self.gray_img[i][j + 1] / 9 + \
                self.gray_img[i + 1][j - 1] / 9 + self.gray_img[i + 1][j] / 9 + self.gray_img[i + 1][j + 1] / 9

        return point

    def linear_smooth_np(self):
        copy_img = self.copy_img()
        for i in range(1, self.img_h - 1):
            for j in range(1, self.img_w - 1):
                copy_img[i][j] = self.done_linear(i, j)

        plt.subplot(121)
        plt.imshow(self.gray_img, 'gray')
        plt.subplot(122)
        plt.imshow(copy_img, 'gray')
        plt.show()

    # 均值滤波
    def linear_smooth(self):
        img_l = cv2.blur(self.img_t, (3, 3))

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(img_l)
        plt.show()

    def box_smooth(self):
        img_b = cv2.boxFilter(self.img_t, -1, (3, 3), normalize=1)

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(img_b)
        plt.show()

    def gaussian_smooth(self):
        img_g = cv2.GaussianBlur(self.img_t, (5, 5), 0)

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(img_g)
        plt.show()

    def median_smooth(self):
        img_m = cv2.medianBlur(self.img_t, 5)

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.subplot(122)
        plt.imshow(img_m)
        plt.show()

    def done_median(self, i, j, c):
        ret = []
        for m in range(-c, c + 1):
            ret.append(self.gray_img[i + m][j])
            if m == 0:
                continue
            ret.append(self.gray_img[i][j + m])
        ret.sort()
        return ret[2 * c]

    def median_smooth_x(self, n):
        c = n // 2
        copy_img = self.copy_img()
        for i in range(c, self.img_h - c):
            for j in range(c, self.img_w - c):
                copy_img[i][j] = self.done_median(i, j, c)

        plt.subplot(121)
        plt.imshow(self.gray_img, 'gray')
        plt.subplot(122)
        plt.imshow(copy_img, 'gray')
        plt.show()
