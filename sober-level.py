import sys
import os

import cv2
import numpy as np

img_fn = sys.argv[1]
bn, en = os.path.splitext(img_fn)
dst_fn = bn + "_sober_edge-level-his" + en

# 读取图像（建议为灰度图）
img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED) 
print("shape:" + str(img.shape))
print("dtype:" + str(img.dtype))
if len(img.shape) >= 3 and img.shape[2] > 1:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图
else:
    gray = img

blur1 = cv2.GaussianBlur(gray, (35, 35), 0)

# 计算 Sobel 梯度（x 和 y 方向）
sobel_x = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur1, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值（与 ImageJ "Find Edges" 一致）
sobel_img = cv2.magnitude(sobel_x, sobel_y)

#低于某阈值的部分清空
sobel_img[sobel_img<40] = 0


ret_img = sobel_img 
# 转换为可显示的 8 位图像
edges = cv2.convertScaleAbs(ret_img)

cv2.imwrite(dst_fn, edges)
