# author:liu ender

from histogram import Histogram
from smoothing import Smooth
from img_change import Change
from base import Base
from img_threshoud import Binary
from dilation_and_erosion import D_E
from warp import Warp
from cvtcolor import Cvt
from edge_detection import Edge_detection
from segmentation import Segmentation
from mosaic import Mosaic
from sampling import Sampling
from image_fusion import Fusion

path1 = "/Users/liumeiyu/Downloads/IMG_7575.JPG"
path2 = "/Users/liumeiyu/Downloads/test1.jpg"
path3 = "/Users/liumeiyu/Downloads/test2.jpg"

A = Histogram(path1)
B = Smooth(path1)
C = Change(path1)
D = Base(path1)
E = Binary(path1)
F = D_E(path1)
G = Warp(path1)
H = Cvt(path1)
K = Edge_detection(path1)
L = Segmentation(path1)
M = Mosaic(path3)
N = Sampling(path1)
P = Fusion(path1)

# A.img_histogram_trans()
# A.img_histogram()

# B.linear_smooth_np()
# B.linear_smooth()
# B.box_smooth()
# B.gaussian_smooth()
# B.median_smooth()
# B.median_smooth_x(5)

# C.fft_high_change(60)
# C.change_cv()

# D.change3()

e = E.get_thrd(5)
# E.biny(e)
print(e)

# F.dilation()
# F.erosion(5)
# F.morph()
# F.morph_3d()

# G.warpA()

# H.change_color()
# H.change_gray()

# K.egde()

# L.k_means()

# M.mosaic()

# N.up_sampling()

# P.img_fusion()
