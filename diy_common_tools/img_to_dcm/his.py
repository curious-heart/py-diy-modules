import numpy as np
import cv2 as cv
import datetime
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

def hist_equalization_single_ch_alt(ori_img):
    #consume too many memory...
    L, type_func = img_dtype_check(ori_img.dtype)
    assert L > 0 and type_func, _img_dtype_range_str
    if 256 == L: 
        start_time = datetime.datetime.now()
        tgt = cv.equalizationHist(ori_img)
        end_time = datetime.datetime.now()
        print("used time: " + str(end_time - start_time))
        return tgt

    start_time = datetime.datetime.now()
    r_list = [ori_img.copy() for i in range(L)]
    for i in range(L): r_list[i][np.logical_not(ori_img == i)] = 0 
    cnt_list = [np.non_zero(r_list[r]) for r in range(L-1, -1, -1)].reverse
    coe_list = [0 for i in range(L)]
    coe_list[L-1] = ori_img.shape[0] * ori_img.shape[1]
    for i in range(L-1, -1, -1): coe_list[i] = coe_list[i+1] - cnt_list[i+1]
    tgt = [np.multiply(r_list[i], coe_list[i]) for i in range(L)]
    end_time = datetime.datetime.now()
    print("used time: " + str(end_time - start_time))
    return np.sum(tgt)


def hist_equalization_single_ch(ori_img):
    L, type_func = img_dtype_check(ori_img.dtype)
    assert L > 0 and type_func, _img_dtype_range_str
    if 256 == L: return cv.equalizationHist(ori_img)

    start_time = datetime.datetime.now()
    print("start:\t" + str(start_time))

    coeffient = (L - 1) / (ori_img.shape[0] * ori_img.shape[1])
    le_r_cnt = 0
    tgt_img = np.zeros_like(ori_img, dtype = np.float64)
    for r in range(L):
        mask = ori_img == r
        le_r_cnt += np.count_nonzero(mask)
        tgt_img[mask] = coeffient * le_r_cnt

    tgt_img = type_func(tgt_img)

    end_time = datetime.datetime.now()
    print("end:\t" + str(end_time))
    print("used time: " + str(end_time - start_time))
    return tgt_img

def hist_equalization(ori_img):
    L, type_func = img_dtype_check(ori_img.dtype)
    assert L > 0 and type_func, _img_dtype_range_str

    hist_eq_chs = []
    for img_ch in cv.split(ori_img):
        hist_eq_chs.append(hist_equalization_single_ch(img_ch))
    return cv.merge(hist_eq_chs)

if len(sys.argv) < 2:
    print("用法：his.py 图像文件名")
    sys.exit(0)

img_fpn = sys.argv[1]
fpn, ext = os.path.splitext(img_fpn)
tgt_fpn = fpn + ".tiff"
ori_img = cv.imread(img_fpn, cv.IMREAD_UNCHANGED)
eq_img = hist_equalization(ori_img)
#eq_img = hist_equalization_single_ch_alt(ori_img)
cv.imwrite(tgt_fpn, eq_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
