import os
import sys
import cv2 as cv
import numpy as np

def exit_prog(code, usage_str = ""):
    """
    code: sys.exit code.
    usage_str: string printed out. can be NULL.
    """
    if usage_str: print(usage_str)
    sys.exit(code)

def split_fpn(pfn):
    """
    parameters:
        pfn: file name with path.

    return dir_name, base_name, ext
    dir_name has \ as tail
    """
    d, ext = os.path.splitext(pfn)
    dir_name = os.path.dirname(d)
    base_name = os.path.basename(d)
    if '' == dir_name:
        dir_name = os.path.abspath(os.path.curdir)
    dir_name += '\\'
    return dir_name, base_name, ext

def read_ch_path_img(img_fn):
    """
    This function load image file whose name may contain Chinese character.
    Parameters:
        img_fn: full path name of image file, may contain Chinese character.
    """
    return cv.imdecode(np.fromfile(img_fn, dtype=np.uint8), cv.IMREAD_UNCHANGED)

def write_ch_path_img(img_fn, img, ext = '.jpg'):
    """
    This function save image file whose name may contain Chinese character.
    Parameters:
        img_fn: full path name of image file, may contain Chinese character.
        img: image data to write.
        ext: image name extension name, .jpg as default.
    """
    ret, img_arr = cv.imencode(ext, img)
    img_arr.tofile(img_fn)
    return ret
    
def add_apx_to_bn(fpn, apx):
    fp_bn, ext = os.path.splitext(fpn)
    return fp_bn + apx + ext

def mkdir_avoid_dup(path_pre, curr_path) -> str:
    """
    基于full_path_base ( = path_pre + "\\" + curr_path) 创建文件夹。如果有重复的，在后面添加"_000“后缀。
    path_pre结尾不带字符"\"。
    后缀最大为999（MAX_MKDIR_TRY - 1）。
    返回生成的curr_path路径名称。
    """
    MAX_MKDIR_TRY = 1000
    full_path_base = path_pre + "\\" + curr_path
    if not hasattr(mkdir_avoid_dup, 'path_cnt_rec'): mkdir_avoid_dup.path_cnt_rec = dict()
    if not (full_path_base in mkdir_avoid_dup.path_cnt_rec.keys()):
        mkdir_avoid_dup.path_cnt_rec[full_path_base] = 0

    apx_format_str = '{:03d}' if "" == curr_path else '_{:03d}'
    cnt_apx = mkdir_avoid_dup.path_cnt_rec[full_path_base]
    output_full_path = full_path_base if 0 == cnt_apx else full_path_base + apx_format_str.format(cnt_apx)
    cnt_apx = (cnt_apx + 1) % MAX_MKDIR_TRY
    while(cnt_apx < MAX_MKDIR_TRY and os.path.exists(output_full_path)):
        output_full_path = full_path_base + apx_format_str.format(cnt_apx)
        cnt_apx += 1

    if cnt_apx >= MAX_MKDIR_TRY:
        cnt_apx = 0
        output_full_path = full_path_base
    mkdir_avoid_dup.path_cnt_rec[full_path_base] = cnt_apx
    if(os.path.exists(output_full_path)):
        shutil.rmtree(output_full_path)
        time.sleep(0.5)
    os.mkdir(output_full_path)
    return output_full_path[output_full_path.rfind("\\") + 1:]
