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

if not(len(sys.argv) in range(3,5)):
    print("用法:\n\timage_range.py raw文件名 灰度值范围 [remap to 灰度值范围]")
    print("\t灰度值范围 格式为：(最小值,最大值)")
    sys.exit(0)

img_file_fpn, val_range_str = sys.argv[1:3]
val_range = str2tuple(val_range_str)
remap_val_range_str = sys.argv[3] if len(sys.argv) >= 4 else ""
fpn, ext = os.path.splitext(img_file_fpn)
tgt_fpn = fpn + "_in_" + str(val_range[0]) + "_" + str(val_range[1])

img = cv.imread(img_file_fpn, cv.IMREAD_UNCHANGED)
L, type_func = img_dtype_check(img.dtype)
assert L > 0 and type_func, _img_dtype_range_str

mask = np.logical_and(val_range[0] <= img, img <= val_range[1])
tgt_img = img.copy()
tgt_img[np.logical_not(mask)] = 0
cv.imwrite(tgt_fpn + ".tiff", tgt_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
if remap_val_range_str:
    remap_val_range = str2tuple(remap_val_range_str)
    tmp = np.float64(tgt_img) - np.min(tgt_img)
    tmp[tmp < 0] = 0
    tmp *= (np.float64((remap_val_range[1] - remap_val_range[0])) / np.float64(np.max(tgt_img) - np.min(tgt_img)))
    tgt_img = type_func(tmp)
    tgt_fpn += "_remapto_" + str(remap_val_range[0]) + "_" + str(remap_val_range[1])

tgt_fpn +=  ".tiff"
cv.imwrite(tgt_fpn, tgt_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
