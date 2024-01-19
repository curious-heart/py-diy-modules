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

if not(len(sys.argv) == 3):
    print("usage: remap.py img_file_name (min, max)")
    sys.exit(0)

img_file_fpn, remap_val_range_str = sys.argv[1:3]
fpn, _ = os.path.splitext(img_file_fpn)
remap_val_range = str2tuple(remap_val_range_str)
if remap_val_range[0] >= remap_val_range[1]:
    print("error: low value must be great than up value.")
    sys.exit(0)
tgt_fpn = fpn + "_remapto_" + str(remap_val_range[0]) + "_" + str(remap_val_range[1]) + ".tiff"

img = cv.imread(img_file_fpn, cv.IMREAD_UNCHANGED)
L, type_func = img_dtype_check(img.dtype)
assert L > 0 and type_func, _img_dtype_range_str

ori_min, ori_max = np.min(img), np.max(img)
tmp = remap_val_range[0] \
    + np.float64(img - ori_min) * (np.float64((remap_val_range[1] - remap_val_range[0])) / np.float64(ori_max - ori_min))
tgt_img = type_func(tmp)
cv.imwrite(tgt_fpn, tgt_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
