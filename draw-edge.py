import cv2
import numpy as np
import sys
import os

# 读取图像
img_fn = sys.argv[1]  # 输入图像路径
img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)  # 保持原位深

if img is None:
    print("无法读取图像")
    sys.exit(1)

print("图像信息:", img.shape, img.dtype)

# 转灰度（若不是单通道）
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img

#-------------------------
# 阈值处理（二值化）
#-------------------------
# 自动确定阈值使用Otsu（注意：Otsu需要8位输入）
if gray.dtype != np.uint8:
    # 将16位数据缩放到0~255再做阈值
    gray_8bit = cv2.convertScaleAbs(gray, alpha=255.0/np.max(gray))
else:
    gray_8bit = gray

# Otsu二值化
_, binary = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#-------------------------
# 形态学操作（先膨胀后腐蚀 = 闭运算，去小黑点）
#-------------------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

#-------------------------
# 提取边界
#-------------------------
# 方法1：膨胀-原图，得到边缘
dilated = cv2.dilate(morph, kernel, iterations=1)
edges = cv2.subtract(dilated, morph)

# 方法2（可选）：cv2.Canny 也能得到轮廓
# edges = cv2.Canny(morph, 50, 150)

#-------------------------
# 保存结果
#-------------------------
bn, ext = os.path.splitext(img_fn)
cv2.imwrite(bn + "_binary" + ext, binary)
cv2.imwrite(bn + "_morph" + ext, morph)
cv2.imwrite(bn + "_edges" + ext, edges)

print("处理完成，已输出结果图像。")
