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

if not(len(sys.argv) in range(2,4)):
    print("用法:\n\tinvert_image.py 图像文件名 [minmax]")
    sys.exit(0)

img_file_fpn = sys.argv[1]
minmax = True if (len(sys.argv) == 3) and ('minmax' == sys.argv[2]) else False
fpn, ext = os.path.splitext(img_file_fpn)
tgt_fpn = fpn + "_invert" + ("_minmax" if minmax else "") + ".tiff"

img = cv.imread(img_file_fpn, cv.IMREAD_UNCHANGED)
L, type_func = img_dtype_check(img.dtype)
assert L > 0 and type_func, _img_dtype_range_str
offset_v = (np.float64(np.max(img)) + np.float64(np.min(img))) if minmax else L -1 
tgt_img = type_func(offset_v - img)
cv.imwrite(tgt_fpn, tgt_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
