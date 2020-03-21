# --->
# Created by liumeiyu on 2020/3/14.
# '_'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from graphy import Graph

'''空域至频域的高低通滤波'''


class Change(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)

    # 任何连续周期信号都可以表示成（或者无限逼近）一系列正弦信号的叠加
    def change_np(self):
        f = np.fft.fft2(self.gray_img)
        fs = np.fft.fftshift(f)
        f_img = np.log(np.abs(fs))  # fft结果是复数, 其结果的绝对值是振幅

        plt.imshow(f_img, 'gray')
        plt.show()

    def change_cv(self):
        dft = cv2.dft(np.float32(self.gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_s = np.fft.fftshift(dft)
        f_img = np.log(cv2.magnitude(dft_s[:, :, 0], dft_s[:, :, 1]))  # 将傅里叶变换的双通道结果(实部和虚部)转换为0到255的范围

        plt.imshow(f_img, 'gray')
        plt.show()

    def fft_high_change(self, d):
        # 傅立叶变换
        f = np.fft.fft2(self.gray_img)
        fshift = np.fft.fftshift(f)

        # 设置高通滤波器,提取图像的边缘轮廓
        crow, ccol = self.img_h // 2, self.img_w // 2
        fshift[crow - d:crow + d, ccol - d:ccol + d] = 0

        # 傅立叶反变换
        ishift = np.fft.ifftshift(fshift)
        iimg = np.fft.ifft2(ishift)
        iimg = np.abs(iimg)

        plt.subplot(121)
        plt.imshow(self.img_t)
        plt.title("ori")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(iimg, cmap='gray')
        plt.title("fourier")
        plt.axis("off")
        plt.show()

    def fft_low_change(self, d):
        dft = cv2.dft(np.float32(self.gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 转换为float32
        fshift = np.fft.fftshift(dft)  # 将图像中的低频部分移动到图像的中心

        crow, ccol = self.img_h // 2, self.img_w // 2
        mask = np.zeros((self.img_h, self.img_w, 2), dtype=np.uint8)  # 实部虚部二维
        mask[crow - d:crow + d, ccol - d:ccol + d] = 1

        f = mask * fshift  # 卷积

        ishift = np.fft.ifftshift(f)
        iimg = cv2.idft(ishift)
        res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])  # 将图像的实部和虚部转换为空间域内

        plt.subplot(121)
        plt.imshow(self.gray_img, 'gray')  # 三通道显示颜色，灰色图需加参数cmap
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(res, 'gray')
        plt.axis('off')

        plt.show()
