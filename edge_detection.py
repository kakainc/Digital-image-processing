# --->
# Created by liumeiyu on 2020/3/18.
# '_'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''边缘检测算法主要是基于图像像素的一阶和二阶导数，但导数通常对噪声很敏感，因此需要采用滤波器（低通）来过滤噪声，并调用图像增强（平滑）或阈值化算（二值化）法进行处理，最后再进行边缘检测。'''


class Edge_detection(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    def egde(self):
        # Roberts交叉微分算法
        kernel_x_r = np.array([[-1, 0], [0, 1]], dtype=np.int)
        kernel_y_r = np.array([[0, -1], [1, 0]], dtype=np.int)

        # Prewitt算子
        kernel_x_p = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernel_y_p = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

        kernel_x = [kernel_x_r, kernel_x_p]
        kernel_y = [kernel_y_r, kernel_y_p]
        edge_name = ['Roberts', 'Prewitt']

        plt.imshow(self.img_t)
        plt.title('original')
        plt.show()

        for i in range(2):
            x = cv2.filter2D(self.img_t, cv2.CV_16S, kernel_x[i])
            y = cv2.filter2D(self.img_t, cv2.CV_16S, kernel_y[i])

            # scaling, taking an absolute value, conversion to an unsigned 8-bit type
            abs_x = cv2.convertScaleAbs(x)
            abs_y = cv2.convertScaleAbs(y)

            img_e = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

            # plt.subplot(222 + i)
            plt.imshow(img_e)
            plt.title(edge_name[i])

            plt.show()

        # Sobel算子
        # kernel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
        # kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)

        x_sb = cv2.Sobel(self.gray_img, cv2.CV_16S, 1, 0)  # ksize表示Sobel算子的大小，其值必须是正奇数
        y_sb = cv2.Sobel(self.gray_img, cv2.CV_16S, 0, 1)

        # Scharr算子
        # kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=int)
        # kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=int)

        x_sc = cv2.Scharr(self.gray_img, cv2.CV_32F, 1, 0)  # 没有ksize参数（默认为3）
        y_sc = cv2.Scharr(self.gray_img, cv2.CV_32F, 0, 1)

        x = [x_sb, x_sc]
        y = [y_sb, y_sc]
        edge_name = ['Sobel', 'Scharr']

        for i in range(2):
            abs_x = cv2.convertScaleAbs(x[i])
            abs_y = cv2.convertScaleAbs(y[i])
            img_es = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

            plt.imshow(img_es, 'gray')
            plt.title(edge_name[i])
            plt.show()

        # Laplacian算子
        dst = cv2.Laplacian(self.gray_img, cv2.CV_16S, ksize=5)
        img_l = cv2.convertScaleAbs(dst)

        plt.imshow(img_l, 'gray')
        plt.title('Laplacian')
        plt.show()

        # Canny算子
        # 1.使用高斯平滑（如公式所示）去除噪声
        # 2.按照Sobel滤波器步骤计算梯度幅值和方向，寻找图像的强度梯度
        # 3.通过非极大值抑制过滤掉非边缘像素，将模糊的边界变得清晰。
        # 4.利用双阈值方法来确定潜在的边界,图像中的像素点如果大于阈值上界则认为必然是边界(称为强边界),
        #   小于阈值下界则认为必然不是边界，两者之间的则认为是候选项(称为弱边界)
        # 5.利用滞后技术来跟踪边界,若某一像素位置的弱边界和强边界相连则认为是边界,其他的弱边界则被删除
        img_gau = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
        img_cny = cv2.Canny(img_gau, 50, 100)  # apertureSize表示Sobel算子大小，默认为3

        plt.imshow(img_cny, 'gray')
        plt.title('Canny')
        plt.show()

        # LOG算子
        # Gauss平滑滤波器和Laplacian锐化滤波器结合

        img_ga = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
        dst_l = cv2.Laplacian(img_ga, cv2.CV_16S, ksize=5)
        img_log = cv2.convertScaleAbs(dst_l)

        plt.imshow(img_log, 'gray')
        plt.title('LOG')
        plt.show()
