# --->
# Created by liumeiyu on 2020/3/10.
# '_'

from graphy import Graph
import matplotlib.pyplot as plt

'''图像的直方图均衡化'''


class Histogram(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def img_histogram(self):
        ret = [0] * 256
        for i in range(self.img_h):
            for j in range(self.img_w):
                ret[self.gray_img[i][j]] += 1  # 返回图像的像素个数直方图

        plt.plot(ret)  # 灰度折线图
        plt.show()

        # hist1 = cv2.calcHist([self.gray_img], [0], None, [256], [0, 255])  # B or G
        # hist2 = cv2.calcHist([self.img], [1], None, [256], [0, 255])       # G
        # hist3 = cv2.calcHist([self.img], [2], None, [256], [0, 255])       # R
        #
        # plt.plot(hist1, c='black')
        # plt.plot(hist2)
        # plt.plot(hist3)
        plt.show()

        return ret

    def img_histogram_trans(self):

        ret = self.img_histogram()

        for i in range(256):
            ret[i] = ret[i] / self.gray_img.size

        for i in range(1, 256):
            ret[i] = ret[i] + ret[i - 1]

        for i in range(256):
            ret[i] = int(ret[i] * 255 + 0.5)

        h_img = self.copy_img()
        for i in range(self.img_h):
            for j in range(self.img_w):
                h_img[i][j] = ret[self.gray_img[i][j]]

        plt.hist(h_img.ravel(), 256)  # 直方图
        plt.show()

        return h_img
