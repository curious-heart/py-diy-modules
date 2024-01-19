import os
import numpy as np
import cv2 as cv
import sys
import os

_img_dtype_range_str = 'Image data can only be uint8 or uint16.'
def img_dtype_check(t):
    if t == 'uint8':
        L = 256
        type_func = np.uint8
    elif t == 'uint16':
        L = 65536
        type_func = np.uint16
    else:
        L = 0
        type_func = ""

    return (L, type_func)

def str2tuple(s):
    """
    input: a string like '(3,25,44)'
    return: a tuple (3,25,44)
    """
    t = s.replace(' ', '')[1:-1].split(',')
    return tuple(int(i) for i in t)

_usage_string = \
"""
usage: sobel.py img_file_name [-so]
"""
if not(len(sys.argv) in range(2,4)):
    print(_usage_string)
    sys.exit(0)

dx = 1
dy = 1
ksize = 3
img_file_fpn = sys.argv[1]
statistic_only = (len(sys.argv) == 3 and '-so' == sys.argv[2])
fpn, ext = os.path.splitext(img_file_fpn)
tgt_fpn = fpn + "_sobel_" + str(dx) + "_" + str(dy) + ".tiff"
sharp_tgt_fpn = fpn + "_sobel_" + str(dx) + "_" + str(dy) + "_sharp.tiff"

img = cv.imread(img_file_fpn, cv.IMREAD_UNCHANGED)
L, type_func = img_dtype_check(img.dtype)
assert L > 0 and type_func, _img_dtype_range_str
tmp = np.float64(img)
dst = np.zeros_like(img)
sobel_x = np.abs(cv.Sobel(tmp, cv.CV_64F, dx, 0, dst, ksize))
sobel_y = np.abs(cv.Sobel(tmp, cv.CV_64F, 0, dy, dst, ksize))
sobel_ret = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
sharp = tmp + sobel_ret
sobel_ret = type_func(sobel_ret)
sharp = type_func(sharp)

if not statistic_only:
    cv.imwrite(tgt_fpn, sobel_ret, (cv.IMWRITE_TIFF_COMPRESSION, 1))
    cv.imwrite(sharp_tgt_fpn, sharp, (cv.IMWRITE_TIFF_COMPRESSION, 1))

print("sobel result statistic:")
print("min:\t" + str(np.min(sobel_ret)))
print("max:\t" + str(np.max(sobel_ret)))
print("sum:\t" + str(np.sum(sobel_ret)))
print("average:\t" + str(np.average(sobel_ret)))
