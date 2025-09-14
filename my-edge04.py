import sys
import os

import numpy as np
import cv2

def detect_boundaries_block_4dir(img, bdr_width=5, slice_width=4, bdr_diff_r=0.4):
    """
    四方向扫描小方块边界：
    - 左→右: left / center < bdr_diff → left_bdr
    - 右→左: right / center < bdr_diff → right_bdr
    - 上→下: bottom / center  < bdr_diff → top_bdr
    - 下→上: top / center < bdr_diff → bottom_bdr
    边界内外全部置0，边界小方块置平均值
    """
    if len(img.shape) >= 3 and img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图
    else:
        gray = img

    dtype = gray.dtype
    h, w = gray.shape
    maxv = 255 if dtype == np.uint8 else 65535

    img_f = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    # ----------------- 左→右 -----------------
    for y0 in range(0, h, slice_width):
        y1 = min(y0 + slice_width, h)
        row_set = False
        for x0 in range(0, w, bdr_width):
            if row_set:
                break
            x1 = min(x0 + bdr_width, w)
            center_block = img_f[y0:y1, x0:x1]
            center_mean = np.mean(center_block)

            # 左邻块
            xl0 = max(0, x0 - bdr_width)
            xl1 = x0
            left_mean = np.mean(img_f[y0:y1, xl0:xl1]) if xl1 > xl0 else 0

            #if center_mean - left_mean > bdr_diff:
            if xl1 > xl0 and center_mean > 0 and  (left_mean / center_mean) < bdr_diff_r:
                out[y0:y1, x0:x1] = center_mean
                out[y0:y1, :x0] = 0
                out[y0:y1, x1:] = 0
                row_set = True

    # ----------------- 右→左 -----------------
    for y0 in range(0, h, slice_width):
        y1 = min(y0 + slice_width, h)
        row_set = False
        for x0 in range(w - bdr_width, -1, -bdr_width):
            if row_set:
                break
            x1 = min(x0 + bdr_width, w)
            center_block = img_f[y0:y1, x0:x1]
            center_mean = np.mean(center_block)

            # 右邻块
            xr0 = x1
            xr1 = min(w, x1 + bdr_width)
            right_mean = np.mean(img_f[y0:y1, xr0:xr1]) if xr1 > xr0 else 0

            #if center_mean - right_mean > bdr_diff:
            if xr1 > xr0 and center_mean > 0 and (right_mean / center_mean) < bdr_diff_r:
                out[y0:y1, x0:x1] = center_mean
                out[y0:y1, :x0] = 0
                out[y0:y1, x1:] = 0
                row_set = True

    # ----------------- 上→下 -----------------
    for x0 in range(0, w, bdr_width):
        x1 = min(x0 + bdr_width, w)
        col_set = False
        for y0 in range(0, h, slice_width):
            if col_set:
                break
            y1 = min(y0 + slice_width, h)
            center_block = img_f[y0:y1, x0:x1]
            center_mean = np.mean(center_block)

            # 下邻块
            yb0 = min(h, y1)
            yb1 = min(h, y1 + slice_width)
            bottom_mean = np.mean(img_f[yb0:yb1, x0:x1]) if yb1 > yb0 else 0

            #if center_mean - bottom_mean > bdr_diff:
            if yb1 > yb0 and center_mean > 0 and (bottom_mean / center_mean) < bdr_diff_r:
                out[y0:y1, x0:x1] = center_mean
                out[:y0, x0:x1] = 0
                out[y1:, x0:x1] = 0
                col_set = True

    # ----------------- 下→上 -----------------
    for x0 in range(0, w, bdr_width):
        x1 = min(x0 + bdr_width, w)
        col_set = False
        for y0 in range(h - slice_width, -1, -slice_width):
            if col_set:
                break
            y1 = min(y0 + slice_width, h)
            center_block = img_f[y0:y1, x0:x1]
            center_mean = np.mean(center_block)

            # 上邻块
            yt0 = max(0, y0 - slice_width)
            yt1 = y0
            top_mean = np.mean(img_f[yt0:yt1, x0:x1]) if yt1 > yt0 else 0

            #if center_mean - top_mean > bdr_diff:
            if yt1 > yt0 and center_mean > 0 and (top_mean / center_mean < bdr_diff_r):
                out[y0:y1, x0:x1] = center_mean
                out[:y0, x0:x1] = 0
                out[y1:, x0:x1] = 0
                col_set = True

    # 转回原位深
    if dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = np.clip(out, 0, 65535).astype(np.uint16)

    return out

img_fn = sys.argv[1]
bn, en = os.path.splitext(img_fn)
dst_fn = bn + "my-edge04" + en

img = cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)  # 8/16位
boundary = detect_boundaries_block_4dir(img, bdr_width=20, slice_width=20, bdr_diff_r = 0.7)
edges = cv2.convertScaleAbs(boundary)
cv2.imwrite(dst_fn, edges)
