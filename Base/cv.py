# author:liu ender

# 导入图像
import cv2
import os
import numpy as np

'''图像基本信息以及显示图像'''


def img_demo(img_path):
    img = cv2.imread(img_path)
    print(type(img), img.size, img.shape, img.dtype)
    print(img)

    cv2.namedWindow('dunk', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('dunk', 300, 100)

    # 单张显示
    cv2.imshow('dunk', img)

    # 多张显示（保证维度相同）
    imgt = np.hstack([img, img])
    cv2.imshow("two_dunk", imgt)

    cv2.waitKey(0)  # 按任何键都会退出显示
    cv2.destroyAllWindows()


'''保存图像'''


def save_demo(in_file, to_file):
    img = cv2.imread(in_file)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(to_file, "gdunk.png"), img_)


'''调用摄像头'''


def video_demo():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)  # 反转图像
        cv2.imshow("video", frame)
        c = cv2.waitKey(50)
        if c == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "/Users/liumeiyu/Downloads/IMG_7575.JPG"
    img_demo(path)
