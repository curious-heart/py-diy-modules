import numpy as np
import cv2 as cv
import datetime
import sys
import os

_img_dtype_range_str = 'Image data can only be uint8 or uint16.'

def exit_prog(code, usage_str = ""):
    """
    code: sys.exit code.
    usage_str: string printed out. can be NULL.
    """
    if usage_str: print(usage_str)
    sys.exit(code)

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

_param_err_str = "invalid parameters. please check the func doc string."
_range_doc_str = \
"""
work_range: str
    'full': (0, 255) or (0, 65535)
    'minmax': (min, max)
    '(w_low, w_high)'
out_range_mode:str
    'as': leave it alone
    'zero': all 0
    'max': all 255 or 65535
    '(below_v, above_v)'
"""
_hist_eq_func_doc = "ori_img: ndarray" + _range_doc_str
def hist_equalization_single_ch(ori_img, work_range, out_range_mode):
    L, type_func = img_dtype_check(ori_img.dtype)
    assert L > 0 and type_func, _img_dtype_range_str

    if (not work_range) or ('full' == work_range): w_low, w_high = 0, L-1
    elif 'minmax' == work_range: w_low, w_high = np.min(ori_img), np.max(ori_img)
    else: [w_low, w_high]= str2tuple(work_range)[0:2]

    leave_out_range_as = False
    if (not out_range_mode) or ('as' == out_range_mode): leave_out_range_as = True
    elif 'zero' == out_range_mode: below_v = above_v = 0
    elif 'max' == out_range_mode:  below_v = above_v = L - 1
    else: [below_v, above_v] = str2tuple(out_range_mode)[0:2]

    if (256 == L) and (0 == w_low and 255 == w_high): return cv.equalizeHist(ori_img)

    start_time = datetime.datetime.now()
    print("start:\t" + str(start_time))

    range_mask = np.logical_and(w_low <= ori_img, ori_img <= w_high)
    MN = np.count_nonzero(range_mask)
    coeffient = (w_high - w_low) / MN
    if leave_out_range_as:
        tgt_img = np.float64(ori_img.copy())
        tgt_img[range_mask] = 0
    else: 
        tgt_img = np.zeros_like(ori_img, dtype = np.float64)

    le_r_cnt = 0
    for r in range(w_low, w_high+1):
        mask = ori_img == r
        cur_cnt = np.count_nonzero(mask)
        if 0 == cur_cnt: continue
        le_r_cnt += cur_cnt
        tgt_img[mask] = coeffient * le_r_cnt

    if not leave_out_range_as:
        if 0 != below_v: tgt[ori_img < w_low] = below_v
        if 0 != above_v: tgt[ori_img > w_high] = above_v

    tgt_img = type_func(tgt_img)

    end_time = datetime.datetime.now()
    print("end:\t" + str(end_time))
    print("used time: " + str(end_time - start_time))
    return tgt_img
hist_equalization_single_ch.__doc__ = _hist_eq_func_doc

def hist_equalization(ori_img, work_range = 'full', out_range_mode = 'as'):
    L, type_func = img_dtype_check(ori_img.dtype)
    assert L > 0 and type_func, _img_dtype_range_str

    hist_eq_chs = []
    for img_ch in cv.split(ori_img):
        hist_eq_chs.append(hist_equalization_single_ch(img_ch, work_range, out_range_mode))
    return cv.merge(hist_eq_chs)
hist_equalization.__doc__ = _hist_eq_func_doc

if not (len(sys.argv) in range(2, 5)):
    print("usage:\nhis.py image_file_name [work_range] [out_range_mode]")
    print("\t" + _range_doc_str)
    sys.exit(0)

img_fpn = sys.argv[1]
work_range = (sys.argv[2] if len(sys.argv) >= 3 else "")
out_range_mode = (sys.argv[3] if len(sys.argv) >= 4 else "")

fpn, ext = os.path.splitext(img_fpn)
tgt_fpn = fpn + "_histEqu" \
        + (("_" + work_range) if work_range else "")\
        + (("_" + out_range_mode) if out_range_mode else "") \
        + ".tiff"
for ch in (' ', '(', ')'): tgt_fpn = tgt_fpn.replace(ch, '')
tgt_fpn = tgt_fpn.replace(',', '-')
ori_img = cv.imread(img_fpn, cv.IMREAD_UNCHANGED)
eq_img = hist_equalization(ori_img, work_range, out_range_mode)
cv.imwrite(tgt_fpn, eq_img, (cv.IMWRITE_TIFF_COMPRESSION, 1))
