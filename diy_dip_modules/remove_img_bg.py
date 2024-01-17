import numpy as np
import cv2 as cv

import diy_common_tools as dct

def rmbg_prep(ori_img):
    img_rmbg = []
    if len(ori_img.shape) == 3:
        if 3 == ori_img.shape[2]:
            #img_rmbg = np.insert(ori_img, 3, 255, 2) 
            img_rmbg = cv.cvtColor(ori_img, cv.COLOR_RGB2RGBA)

        elif 4 == ori_img.shape[2]:
            img_rmbg = ori_img        
        else:
            #To Be Filled
            pass
    else:
        #To Be filled
        pass
    return img_rmbg

_RMBG_APX_STR = '_rmbg'
_WHITE_3P = (255, 255, 255)
def rm_img_bg_by_bgcolor(fpn, bgc = _WHITE_3P, write_file = True):
    """
    Remove the "bgc" color background area of an image, that is, make its background "transparent".
    "bgc" can be of two formats:
    1) a single color: (r, g, b)
    2) a list of colors: ((r1, g1, b1), (r2, g2, b2)...)
    Currently, the output file can only be PNG format.
    """
    ori_img = dct.read_ch_path_img(fpn)
    img_rmbg = rmbg_prep(ori_img)
    if len(img_rmbg) == 0:
        return None

    bgc_list = []
    bgc = np.array(bgc)
    if(len(bgc.shape) == 1): #(r,g,b) format
        bgc = np.array((bgc,))
    else:
        pass #assuming ((r1,g1,b1),..) format
    #now, bgc is in shape of (n,3), where n is the number of color.
    #since opencv read file in BGR mode by default, while bgc is in rgb mode, we flip bgc before comparing.
    bgc = np.flip(bgc, axis = 1) 

    indices_arr = np.all(img_rmbg[:, :, 0:3] == bgc[0], axis = 2)
    for i in range(len(bgc) - 1):
        i_arr = np.all(img_rmbg[:, :, 0:3] == bgc[i+1], axis = 2)
        indices_arr = np.logical_or(indices_arr, i_arr)
    img_rmbg[indices_arr, 3] = 0

    if write_file:
        ret_fpn = dct.add_apx_to_bn(fpn, _RMBG_APX_STR)
        ret_fpn = dct.change_ext(ret_fpn, '.png')
        dct.write_ch_path_img(ret_fpn, img_rmbg, '.png')
    return img_rmbg

_DF_D = (5, 5, 5)
def rm_img_bg_by_contour(fpn, df_d = _DF_D, ord = 'r+c', write_file = True):
    """
    ord: 'row', or 'col'
    """
    ori_img = dct.read_ch_path_img(fpn)
    img_rmbg = rmbg_prep(ori_img)
    
    if len(img_rmbg) == 0:
        return None

    row_num, col_num = img_rmbg.shape[0], img_rmbg.shape[1]
    opd_list = ord.split('+')
    for op in opd_list:
        if('r' == op):
            for r in range(row_num):
                rd = np.roll(img_rmbg[r,:,:], 1, 0)
                dif_d = np.abs((np.int32(rd) - np.int32(img_rmbg[r,:,:]))[:,:3])
                dif_d[0] = 0
                flag = np.full(len(dif_d), False)
                for i in range(len(dif_d)):
                    if(all(dif_d[i] <= df_d)): flag[i] = True
                    else: break
                if i < len(dif_d) - 1:
                    for i in range(len(dif_d) - 1, 0, -1):
                        if(all(dif_d[i] <= df_d)): flag[i] = True
                        else:
                            if i > 0: flag[i-1] = True
                            break
                img_rmbg[r, flag, 3] = 0
        elif ('c' == op):
            for c in range(col_num):
                cd = np.roll(img_rmbg[:,c,:], 1, 0)
                dif_d = np.abs((np.int32(cd) - np.int32(img_rmbg[:,c,:]))[:,:3])
                dif_d[0] = 0
                flag = np.full(len(dif_d), False)
                for i in range(len(dif_d)):
                    if(all(dif_d[i] <= df_d)): flag[i] = True
                    else: break
                if i < len(dif_d) - 1:
                    for i in range(len(dif_d) - 1, 0, -1):
                        if(all(dif_d[i] <= df_d)): flag[i] = True
                        else:
                            if i > 0: flag[i-1] = True
                            break
                img_rmbg[flag, c, 3] = 0
    
    if write_file:
        ret_fpn = dct.add_apx_to_bn(fpn, _RMBG_APX_STR)
        ret_fpn = dct.change_ext(ret_fpn, '.png')
        dct.write_ch_path_img(ret_fpn, img_rmbg, '.png')

def rm_img_bg_gray_list(fpn, gray_c_list, delta = 0, write_file = True):
    bgc_list = [(c,c,c) for c in gray_c_list]
    del_part = [(c,c,c+1) for c in gray_c_list]
    bgc_list += del_part

    rm_img_bg_by_bgcolor(fpn, bgc_list, write_file)

def rm_img_bg_gray_range(fpn, gray_s, gray_e, delta = 0, write_file = True):
    bgc_list = [(c,c,c) for c in range(gray_s, gray_e +1)]
    del_part = [(c,c,c+1) for c in range(gray_s, gray_e +1)]
    bgc_list += del_part

    rm_img_bg_by_bgcolor(fpn, bgc_list, write_file)
