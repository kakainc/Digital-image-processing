# --->
# Created by liumeiyu on 2020/3/19.
# '_'

import cv2
from graphy import Graph

'''对选中区域马赛克（采样）'''


class Mosaic(Graph):
    def __init__(self, img_path):
        super().__init__(img_path)
        self.en = False

    def drowmask(self, x, y, size=50):  # size: 影响范围
        for i in range(size):
            for j in range(size):
                self.img[x + i][y + j] = self.img[x][y]

    # onMouse响应函数
    def on_mouse(self, event, x, y, flags, parms):  # setMouseCallback()函数给回调函数传递的参数
        print(parms)
        if event == cv2.EVENT_LBUTTONDOWN:  # 点一下开启
            self.en = True
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
            if self.en:
                print(x, y)
                self.drowmask(y, x)  # 矩阵像素和鼠标获取点(以左上为00，右为x轴)的坐标值相反
            # elif event == cv2.EVENT_LBUTTONUP:
            #     self.en = False

        # global x1, y1
        # if event == cv2.EVENT_LBUTTONDOWN:
        #     x1, y1 = x, y
        # if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        #     cv2.rectangle(self.img, (x1, y1), (x, y), (0, 255, 0), -1)

    def mosaic(self):
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.on_mouse, param='mouse')  # 操作响应

        while True:
            cv2.imshow('img', self.img)
            if cv2.waitKey(1) & 0xFF == 27:  # esc键退出
                break

        # cv2.imshow('img', self.img)
        # if cv2.waitKey() & 0xFF == 27:  # esc键退出
        #     cv2.destroyWindow('img')
        # cv2.imshow('mask_img', self.img)
        # cv2.waitKey(0)

        cv2.destroyAllWindows()

    '''
    Event:
    EVENT_MOUSEMOVE         0 // 滑动
    EVENT_LBUTTONDOWN       1 // 左键点击   down 点击  取点击的点(鼠标移动过程中每一个点)
    EVENT_RBUTTONDOWN       2 // 右键点击
    EVENT_MBUTTONDOWN       3 // 中键点击
    EVENT_LBUTTONUP         4 // 左键放开   up 放开
    EVENT_RBUTTONUP         5 // 右键放开
    EVENT_MBUTTONUP         6 // 中键放开
    EVENT_LBUTTONDBLCLK     7 // 左键双击  blclk 双击
    EVENT_RBUTTONDBLCLK     8 // 右键双击
    EVENT_MBUTTONDBLCLK     9 // 中键双击  
    
    intx, inty代表鼠标位于窗口的（x，y）坐标位置
    
    int flags代表鼠标的拖拽事件，以及键盘鼠标联合事件
    flags:  
    EVENT_FLAG_LBUTTON 1       //左鍵拖曳    取拖拽结尾的点
    EVENT_FLAG_RBUTTON 2       //右鍵拖曳  
    EVENT_FLAG_MBUTTON 4       //中鍵拖曳  
    EVENT_FLAG_CTRLKEY 8       //(8~15)按Ctrl不放事件  
    EVENT_FLAG_SHIFTKEY 16     //(16~31)按Shift不放事件  
    EVENT_FLAG_ALTKEY 32       //(32~39)按Alt不放事件
    '''
