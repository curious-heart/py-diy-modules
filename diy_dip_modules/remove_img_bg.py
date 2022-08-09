import numpy as np
import cv2 as cv

import diy_common_tools as dct

_RMBG_APX_STR = '_rmbg'
_WHITE_3P = (255, 255, 255)
def rm_img_bg(fpn, bgc = _WHITE_3P, write_file = True):
    """
    Remove the "bgc" color background area of an image, that is, make its background "transparent".
    "bgc" can be of two formats:
    1) a single color: (r, g, b)
    2) a list of colors: ((r1, g1, b1), (r2, g2, b2)...)
    Currently, the output file can only be PNG format.
    """
    ori_img = dct.read_ch_path_img(fpn)
    if len(ori_img.shape) == 3:
        if 3 == ori_img.shape[2]:
            #img_rmbg = np.insert(ori_img, 3, 255, 2) 
            img_rmbg = cv.cvtColor(ori_img, cv.COLOR_RGB2RGBA)

        elif 4 == ori_img.shape[2]:
            img_rmbg = ori_img        
        else:
            #To Be Filled
            return None
    else:
        #To Be filled
        return None

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
