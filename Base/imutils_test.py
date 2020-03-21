# --->
# Created by liumeiyu on 2020/3/15.
# '_'

import imutils

import cv2
import matplotlib.pyplot as plt

'''图像的边界骨架提取'''


def auto_canny(img_path):
    img = cv2.imread(img_path, 0)
    img_canny = imutils.auto_canny(img)
    img_skeleton = imutils.skeletonize(img, size=(3, 3))

    plt.subplot(131)
    plt.imshow(imutils.opencv2matplotlib(img_canny))
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(imutils.opencv2matplotlib(img))
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(imutils.opencv2matplotlib(img_skeleton))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    path = "/Users/liumeiyu/Downloads/test2.jpg"
    path2 = "/Users/liumeiyu/Downloads/IMG_7575.JPG"
    auto_canny(path)
