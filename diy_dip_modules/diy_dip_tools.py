import numpy as np
import cv2 as cv
import os
import sys
from collections import abc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

ABS_S = 'abs'
REAL_S = 'real'
IMAG_S = 'imag'
ORI_S = 'ori'
DFT_S = 'dft'
DFT2_S = 'dft2'
IDFT2_S = 'idft2'
ROW_S = 'row'
COL_S = 'col'
DFT2_ROW_S = DFT2_S + ROW_S
DFT2_COL_S = DFT2_S + COL_S
IDFT2_ROW_S = IDFT2_S + ROW_S
IDFT2_COL_S = IDFT2_S + COL_S
DATA_S = 'data'
SPTRUM_IMG_S = 'sptrum_img'
ANGLE_IMG_S = 'angle_img'
SPTRUM_TXT_S = 'sptrum_txt'
ANGLE_TXT_S = 'angle_txt'
FILE_NAME_S = 'file_name'

def gen_sptrum_img(data_array, part = ABS_S, expand = False, identical_trans = False, identical_data = 0):
    """
    Generate 2D array with value 0~255 so that it can be write as image directly.
    Parameters:
        data_array: The data from which the image data is generated. a 2D array like structure. Generally, it is
            output of dft2/idft2.
        part: How to draw data from data_array:
            ABS_S: np.abs(data_array)
            REAL_S: np.real(data_array)
            IMAG_S: np.imag(data_array)
            ORI_S: use data_array directly. With this option, the data_array should contain all real data.
        expand: Boolean, default False. If true, the output data is expanded. Currently, log10 is used.
        idential_trans: Only meaningful when all data drew from data_array are identical: 
            if True, output data are set to value assigned to identical_data; otherwise, output drew data.
        identical_data: Refer to identical_trans.
    Returned:
        2D array with the same shape with data_array, valued 0~255.
    """
    if ABS_S == part:
        data = np.abs(data_array)
    elif REAL_S == part:
        data = np.real(data_array)
    elif IMAG_S == part:
        data = np.imag(data_array)
    else: #ORI_S
        data = np.array(data_array)
    
    if np.max(data) == np.min(data):
        if identical_trans: 
            if identical_data < 0: identical_data = 0
            elif identical_data > 255: identical_data = 255
            data = identical_data
        data = data.astype(np.uint8)
        return data

    data -= np.min(data)
    if expand:#To be complete more styles
       data += 1
       data = np.log10(data)
    data = data / (np.max(data) - np.min(data))
    data *= 255
    data = data.astype(np.uint8)
    return data

def gen_angle_img(dft2_data, expand = False, identical_trans = False, identical_data = 0):
    """
    Generate 2D array with value 0~255 so that it can be write as image directly.
    Parameters:
        dft2_data: input data. Generallyj, it is output of dft2/idft2.
        expand: Boolean, default False. If true, the output data is expanded. Currently, log10 is used.
        idential_trans: Only meaningful when all angle of dft2_data are identical: 
            if True, output data are set to value assigned to identical_data; otherwise, output ori uint8(data).
        identical_data: Refer to identical_trans.
    Returned data can be written as img directly.
    """
    data = np.angle(dft2_data)
    if np.max(data) == np.min(data):
        if identical_trans: 
            if identical_data < 0: identical_data = 0
            elif identical_data > 255: identical_data = 255
            data = identical_data
        else:
            if np.min(data) < 0: data = 0
        data = data.astype(np.uint8)
        return data

    data -= np.min(data)
    if expand:
        data += 1
        data = np.log10(data)
    data = 255 * (data - np.min(data))/(np.max(data) - np.min(data))
    data = data.astype(np.uint8)
    return data

def display_1d2darr(arr, fh = sys.stdout, sep = '\t', new_line = '\n', one_d = False, oned_line_item_number = 0):
    """
    Display 1d or 2d array. Every line (including the last line) ends with new_line.
    Parameters:
        arr: Array to be displayed. If ond_d is True, arr is 1d, otherwise, it is 2d.
        fh: Where to print the array. If None, no print, just return string.
        sep: Seperator between array item for display.
        new_line: What ends a line in display. Line definition: for 2d array, a row(arr[i]) is a line; for 1d array,
            if oned_line_item_number is 0 (default), the whole array is one line; otherwise, the oned_line_item_number
            number of items in array forms a line.
        one_d: True to indicate the arr is 1d, and False to 2d.
        oned_line_item_number: Only meaningful if one_d is True, defining the number of items in a displayed-line.
            Refer to comments in new_line.
    Returned Value:
        A string seperated by sep and new_line than can be print directly.
    """
    str_result = []
    if one_d: 
        cn = len(arr)
        if oned_line_item_number <=0 : oned_line_item_number= cn + 1
        item_number = 0
        for c in range(cn):
            str_result.append(str(arr[c]))
            item_number += 1
            if (item_number == oned_line_item_number) or (c == cn - 1):
                str_result.append(new_line)
                item_number = 0
            else:
                str_result.append(sep)
    else:
        rn = len(arr)
        cn = len(arr[0])
        for r in range(rn):
            for c in range(cn):
                str_result.append(str(arr[r][c]))
                if c < cn -1 : str_result.append(sep)
            str_result.append(new_line)
    str_result = ''.join(str_result)
    if fh != None: print(str_result, file = fh, end = '')
    return str_result

def display_2darr(arr, fh = sys.stdout, sep = '\t', new_line = '\n'):
    """
    Display 2d array. Items are sperated by sep, and every line (including the last line) ends with new_line.
    If fh is None, no print.
    Returned Value:
        A string seperated by sep and new_line than can be print directly.
    """
    return display_1d2darr(arr, fh, sep, new_line, False)

def display_1darr(arr, fh = sys.stdout, sep = '\t', new_line = '\n', oned_line_item_number = 0):
    """
    Display 1d array. Items are sperated by sep, and every line (including the last line) ends with new_line.
    If fh is None, no print.
    If oned_line_item_number is 0, the whole array is displayed as one line; otherwise, the oned_line_item_number
    of items are displayed in one line.
    Returned Value:
        A string seperated by sep and new_line than can be print directly.
    """
    return display_1d2darr(arr, fh, sep, new_line, True, oned_line_item_number)

__SPTRUM_S = 'sptrum'
__ANGLE_S = 'angle'
__NO_SHIFT_S = 'noshift'
__SHIFT_S = 'shift'
__TXT_EXT = '.txt'
__IMG_EXT = '.png'
__FUNC_S = 'func'
__PARM_S = 'parm'
__NAME_APPEND_S = 'append'
__NAME_PREPEND_S = 'prepend'
__gen_img_dft2_func_dict = {
        DFT2_S:{
            __FUNC_S: lambda img, shift: np.fft.fftshift(np.fft.fft2(img, axes = (0,1)),axes = (0,1)) if shift\
                                       else np.fft.fft2(img, axes = (0,1)),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            }, 
        DFT2_ROW_S:{
            __FUNC_S: lambda img, shift: np.fft.fftshift(np.fft.fft(img, axis = 1), axes = 1) if shift\
                                       else np.fft.fft(img, axis = 1),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            },
        DFT2_COL_S:{
            __FUNC_S: lambda img, shift: np.fft.fftshift(np.fft.fft(img, axis = 0), axes = 0) if shift\
                                       else np.fft.fft(img, axis = 0),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            },
        IDFT2_S:{
            __FUNC_S: lambda img, shift: np.fft.ifft2(np.fft.ifftshift(img, axes = (0,1)), axes = (0,1)) if shift\
                                       else np.fft.ifft2(img, axes = (0,1)),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            },
        IDFT2_ROW_S:{
            __FUNC_S: lambda img, shift: np.fft.ifft(np.fft.ifftshift(img, axes = 1), axis = 1) if shift\
                                       else np.fft.ifft(img, axis = 1),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            },
        IDFT2_COL_S:{
            __FUNC_S: lambda img, shift: np.fft.ifft(np.fft.ifftshift(img, axes = 0), axis = 0) if shift\
                                       else np.fft.ifft(img, axis = 0),
            __NAME_APPEND_S: lambda shift: __SHIFT_S if shift else __NO_SHIFT_S,
            __NAME_PREPEND_S: lambda shift: '',
            },
}
def gen_img_dft_datum(img_data, func_str, shift = True,\
        sptrum_part = ABS_S, sptrum_expand = False, angle_expand = False,\
        sptrum_identical_trans = False, sptrum_identical_data = 0,\
        angle_identical_trans = False, angle_identical_data = 0,\
        base_name = '', img_ext = __IMG_EXT, txt_ext = __TXT_EXT, \
        txt_sep = '\t', txt_new_line = '\n', txt_new_layer = '\n'):
    """
    Input an image as img_data, assigning the function as func_str, and output a dictionary, containing the DFT/IDFT
    result, including original data, sptrum/angle image data normalized between 0 and 255 so that it can be written
    as image directly, and text format of sptrum/angl image.

    Parameters:
        img_data: Image data, generally it is the return value of cv2.imread. It is not necessary the img_data is
            2D, it can also be more than 2D, and every "2D layer" is processed the same way.
        func_str: The following options:
            DFT2_S: dft2, that is, on axes (0, 1).
            DFT2_ROW_S: dft on every "row", that is, axis 1.
            DFT2_COL_S: dft on every "column", that is, axis 0.
            IDFT2_S: idft2, that is, on axes (0, 1).
            IDFT2_ROW_S: idft on every "row", that is, axis 1.
            IDFT2_COL_S: idft on every "column", that is, axis 0.
        shift: True if the "shift" operation is taken. For dft operation, img_data is dft first and then shift;
            for idft operation, img_data is shift and then idft.
        sptrum_part: how to drwa sptrum image data:
            ABS_S: np.abs(data)
            REAL_S: np.real(data)
            IMAG_S: np.imag(data)
            ORI_S: data. Only if sptrum data is real can this be used.
        sptrum_expand/angle_expand: if true, the generated sptrum/angle image is expanded. Currently, log10 is taken.
        sptrum_idential_trans/angle_identical_trans: Only meaningful when all sptrum/angle image data are identical: 
            if True, output data are set to value assigned to sptrum/angle_identical_data;
            otherwise, output ori uint8(data).
        sptrum_identical_data/angle_identical_data: Refer to sptrum_identical_trans/angle_identical_trans.
        base_name: base name for generated data. If None, the output file name is null('').
        img_ext/txt_ext: extent name of file name for sptrum/angle image data and text data file.
        txt_sep/txt_new_line: seperator and line-end character(s) for output text file.
        txt_new_layer: layer-end character(s) for output text file. (Layer example: for shape=(x,y,3), there are 3
            layers; for shape=(x,y), there is 1 layer.
    Returned Value:
        An emebded dictionary:
            DATA_S:
                DATA_S: original dft/idft data.
                FILE_NAME_S: file name that can be used for write data file. Generally, np.save can use this string.
            SPTRUM_IMG_S/ANGLE_IMG_S:
                DATA_S: sptrum/angle data normalized between 0 and 255 for write as image.
                FILE_NAME_S: file name that can be used for write 'data' as an image.
            SPTRUM_TXT_S/ANGLE_TXT_S:
                DATA_S: formated sptrum/angle image data for display.
                FILE_NAME_S: file name that can be used for write 'data' as a text file.
        Note: if base_name parameter is null, all 'file_name' in dictionary is null('').
    """
    ori_data = __gen_img_dft2_func_dict[func_str][__FUNC_S](img_data, shift) 
    ori_data = np.array(ori_data)
    if len(ori_data.shape) == 2:
        single_layer = True
        layer_num = 1
        ori_data_layers = ori_data[np.newaxis, :, :]
    else:
        single_layer = False
        layer_num = ori_data.shape[2]
        ori_data_layers = np.split(ori_data, layer_num, -1) #now element in ori_data_layers is (M,N,1)
        ori_data_layers = np.array([np.squeeze(x, axis = -1) for x in ori_data_layers])
    #now, element in ori_data_layers are 2D
    sptrum_img_layers = []
    angle_img_layers = []
    sptrum_txt_layers = []
    angle_txt_layers = []
    for layer in range(layer_num):
        sptrum_img = gen_sptrum_img(ori_data_layers[layer], sptrum_part, sptrum_expand, \
                                    sptrum_identical_trans, sptrum_identical_data)
        angle_img = gen_angle_img(ori_data_layers[layer], angle_expand,\
                                  angle_identical_trans, angle_identical_data)
        sptrum_img_layers.append(sptrum_img)
        angle_img_layers.append(angle_img)
        sptrum_txt_layers.append(display_2darr(sptrum_img, None, txt_sep, txt_new_line))
        angle_txt_layers.append(display_2darr(angle_img, None, txt_sep, txt_new_line)) 
    #keep sptrum_img_data and angle_img_data the same shape with img_data.
    if single_layer:
        sptrum_img_data = sptrum_img_layers[0]
        angle_img_data = angle_img_layers[0]
    else:
        sptrum_img_data = np.stack(sptrum_img_layers, axis = -1)
        angle_img_data = np.stack(angle_img_layers, axis = -1)
    sptrum_txt_data = txt_new_layer.join(sptrum_txt_layers)
    angle_txt_data = txt_new_layer.join(angle_txt_layers)
    if base_name != '':
        name_append = __gen_img_dft2_func_dict[func_str][__NAME_APPEND_S](shift)
        name_prepend = __gen_img_dft2_func_dict[func_str][__NAME_PREPEND_S](shift)
        bn, en = os.path.splitext(base_name)
        data_name = '{}_{}_{}_{}_{}'.format(bn, name_prepend, func_str, name_append, DATA_S) 
        sptrum_img_name = '{}_{}_{}_{}_{}{}'.format(bn, name_prepend, func_str, name_append, SPTRUM_IMG_S, img_ext) 
        angle_img_name = '{}_{}_{}_{}_{}{}'.format(bn, name_prepend, func_str, name_append, ANGLE_IMG_S, img_ext)
        sptrum_txt_name = '{}_{}_{}_{}_{}{}'.format(bn, name_prepend, func_str, name_append, SPTRUM_TXT_S, txt_ext)
        angle_txt_name = '{}_{}_{}_{}_{}{}'.format(bn, name_prepend, func_str, name_append, ANGLE_TXT_S, txt_ext)
    else:
        data_name = ''
        sptrum_img_name = ''
        angle_img_name = ''
        sptrum_txt_name = ''
        angle_txt_name = ''
    result_dict = dict()
    result_dict[DATA_S] = {DATA_S: ori_data, FILE_NAME_S: data_name}
    result_dict[SPTRUM_IMG_S] = {DATA_S: sptrum_img_data, FILE_NAME_S: sptrum_img_name}
    result_dict[ANGLE_IMG_S] = {DATA_S: angle_img_data, FILE_NAME_S: angle_img_name}
    result_dict[SPTRUM_TXT_S] = {DATA_S: sptrum_txt_data, FILE_NAME_S: sptrum_txt_name}
    result_dict[ANGLE_TXT_S] = {DATA_S: angle_txt_data, FILE_NAME_S: angle_txt_name}
    return result_dict

def expand_array(arr_data, dst_dim_size, axes = (0,1), orig_point = (0, 0), filled_value = 0):
    """
    Expand the dimention size along axes of arr_data, filled with filled value. arr_data is originated at orig_point
    of expanded array.
    Parameters:
        arr_data: array like input data.
        dst_dim_size: An integer or a list of intergers, indicating dimention size of axes in the expanded result.
            It SHOULD contain the same number of elements as that in axes.
        axes: An integer or a list of intergers, indicating which axes are to be expanded. It SHOULD contain
            the same number of elements as that in dst_dim_size. The order of indices in axes does not matter, that
            is, (0,1) and (1,0) has the same effect. The default value is (0,1).
        orig_point: An integer or a list of intergers, indicating the origin point of the arr_data in expanded result.
            It should contain the same number of elements as that in axes and dst_dim_size. (0,0) by default.
            Note: The elements in dst_dim_size, axes, orig_point are one-to-one corrdinated in order. That is, the 1st
            element in dst_dim_size is the dimention size of axis indicated by the 1st element of axes, and that axis
            of arr_data is put at the position indicated by the 1st value of orig_point in the expanded result; and the
            2nd elements work like above, and so on.
        filled_value: value to be filled in the expanded positions. 0 by default.
    Returned Value:
        The expanded array of type ndarray. If number of elements in dst_dim_size, axes and orig_point are not of the
        same, None is returned.
    Note: We don't check the validity of axes or orig_point. Let python/numpy build-in index checking do that.
    """
    if not isinstance(dst_dim_size, abc.Iterable): dst_dim_size = [dst_dim_size]
    if not isinstance(axes, abc.Iterable): axes = [axes]
    if not isinstance(orig_point, abc.Iterable): orig_point = [orig_point]

    num_of_expand_axes = len(dst_dim_size)
    if (num_of_expand_axes != len(axes)) or (num_of_expand_axes != len(orig_point)):
        return None
    arr_data = np.array(arr_data)
    dst_shape = list(arr_data.shape)
    slices_list = []
    for i in range(len(dst_shape)):
        if i in list(axes):
            axes_i = axes.index(i)
            dst_shape[i] = dst_dim_size[axes_i]
            curr_slice = slice(orig_point[axes_i], orig_point[axes_i] + arr_data.shape[i])
        else:
            curr_slice = slice(None, None)
        slices_list.append(curr_slice)
    dst_arr = np.full(dst_shape, filled_value)
    dst_arr[tuple(slices_list)] = arr_data
    return dst_arr

_BASE_RCOUNT = 256
_BASE_CCOUNT = 256
def pixels_standup(img_data, form = 'surface', cmap = cm.gray):
    """
    As its name, this function draw a 3D figure from the input 2D figure, with the x and y axis corresponds to
    number of row and column respectively, and z axis refelects the intensity of pixel.
    Parameters:
        img_data: a 2D image. Every "layer" of it is transformed in a 3D figure, e.g., an image of shape (300, 200, 3)
            results 3 3D image, each of the x and y axis corresponding to 300 and 200 direction.
        form: Specify the form of the output 3D figure. Currently two options are available: 'surface' and 'bar'.
        cmap: if form is 'surface', cmap assigns the color map. cm.gray is as default.
    Returns:
        None.
    """
    img_data = np.array(img_data)
    row ,col = img_data.shape[0], img_data.shape[1]
    rc = max(row, _BASE_RCOUNT)
    cc = max(col, _BASE_CCOUNT)
    if len(img_data.shape) > 2: layer = img_data.shape[2]
    else: layer = 1
    _xx, _yy = np.meshgrid(range(row), range(col), indexing = 'ij')
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(x)
    if 1 == layer:
        top = [img_data.ravel()]
        surf = [img_data]
    else:
        top = [img_data[:,:,l].ravel() for l in range(layer)]
        surf = [img_data[:,:,l] for l in range(layer)]
    fig = plt.figure()
    axes = [fig.add_subplot(1, layer, l+1, projection = '3d') for l in range(layer)]
    for l in range(layer):
        axes[l].set_xlabel('row')
        axes[l].set_ylabel('col')
        axes[l].set_xticks(range(row))
        axes[l].set_yticks(range(col))
        if 'bar' == form:
            axes[l].bar3d(x, y, bottom, 1, 1, top[l])
        else: #surface
            c_surf = axes[l].plot_surface(_xx, _yy, surf[l], rcount = rc, ccount = cc, linewidth = 0, cmap=cmap)
            fig.colorbar(c_surf)
    plt.show()

def _gen_2d_wh_sin_fig(rcount, ccount, *, Tr, Tc, Pr = 0, Pc = 0):
    """
    Gen sin(x+y) 2d figure with color of white and black.
    Parameters:
        rcount: Number of row of output figure.
        ccount: Number of column of output figure.
        Tr: Period in row direction.
        Tc: Period in column direction.
        Pr: Intial phase in row direction.
        Pc: Intial phase in column direction.
    Returns:
        The 2D figure in ndarray format.
    The figure is generated use the following formula:
        sin(2*Pi*(x/Tr + y/Tc) + Pr + Pc)
    If Tr is 0, that means a constant value in each row; Tc is the same case.
    """
    Fr = 1/Tr if Tr > 0 else 0
    Fc = 1/Tc if Tc > 0 else 0
    if 0 == Fr: Pr = 0
    if 0 == Fc: Pc = 0
    r_d, c_d = np.meshgrid(range(rcount), range(ccount), indexing = 'ij')
    s_d = 2 * np.pi * (Fr * r_d + Fc * c_d) + Pr + Pc
    ori_d = np.sin(s_d)
    img_d = gen_sptrum_img(ori_d, ORI_S)
    return img_d

def _gen_2d_wh_cos_fig(rcount, ccount, *, Tr, Tc, Pr = 0, Pc = 0):
    """
    Refer to help of _gen_2d_wh_sin_fig.
    """
    return _gen_2d_wh_sin_fig(rcount, ccount, Tr=Tr, Tc=Tc, Pr = Pr + np.pi/2, Pc = Pc + np.pi/2)

def _construct_grid_period(wh, bl, start, start_pos):
    """
    Construct a period of data in a grid pattern.
    Parameters:
        wh: number of white dots.
        bl: number of black dots.
        start: 'white' means white dots start and 'black' means black dots start.
        start_pos: the "phase" of the period: if 0, the period consists of wh white dots and bl black dots, and
            this is called a "normal period"; if not 0, the period starts at the start_pos of a normal period.
    Returns:
        A 1-D ndarray, with "1" indicating white, and "-1" indicating black.
    """
    wh_part = np.full((wh,), 1)
    bl_part = np.full((bl,), -1)
    if 'white' == start:
        part = np.concatenate((wh_part, bl_part))
    else: #'black'
        part = np.concatenate((bl_part, wh_part))
    part = np.tile(part, 2)
    start_pos = start_pos % (wh + bl)
    part = part[start_pos: start_pos + wh + bl]
    return part

def _gen_2d_wh_grid_fig(rcount, ccount, *, 
                       rd_wh, rd_bl, rd_start = 'white', rd_start_pos = 0, 
                       cd_wh, cd_bl, cd_start = 'white', cd_start_pos = 0):
    """
    Generate a 2D grid figure with the color of white and black.
    Parameters:
        rcount/ccount: Number of rows/columns of the output figure.
        rd_wh/rd_bl/cd_wh/cd_bl: The width of white/black in one period in row/column direction ("rd" means
            "row direction", and "cd" means "column direction"; "wh" means "white" and "bl" means "black"). 
        rd_start/cd_start: 'white' means white part is at the beginning, and 'black' means black part is at
            the beginning.
        rd_start_pos/cd_start_pos: the "phase" of the period: if 0, the period consists of rd_wh/cd_wh white
            dots and rd_bl/cd_bl black dots, and this is called a "normal period"; if not 0, the period starts
            at the rd_start_pos/cd_start_pos of a normal period.
    Returns:
        A 2D ndarray of grid. If rd_wh or rd_bl is 0, the output pattern is vertical stripe; if cd_wh or cd_bl
        is 0, the output pattern is horizontal stripe.
    """
    if 0 == rd_wh * rd_bl:
        rd_arr = np.full((rcount, ccount), 1)
    else:
        rd_period = construct_grid_period(rd_wh, rd_bl, rd_start, rd_start_pos)
        rd_cnt = rcount // len(rd_period)
        rd_remain = rcount % len(rd_period)
        rd_bar = np.concatenate((np.tile(rd_period, rd_cnt), rd_period[:rd_remain]))
        rd_arr = np.tile(np.transpose([rd_bar]), ccount)

    if 0 == cd_wh * cd_bl:
        cd_arr = np.full((rcount, ccount), 1)
    else:
        cd_period = construct_grid_period(cd_wh, cd_bl, cd_start, cd_start_pos)
        cd_cnt = ccount // len(cd_period)
        cd_remain = ccount % len(cd_period)
        cd_bar = np.concatenate((np.tile(cd_period, cd_cnt), cd_period[:cd_remain]))
        cd_arr = np.tile(cd_bar, (rcount,1))
    result = rd_arr * cd_arr
    result = result > 0
    result = 255 * result.astype(np.uint8)
    return result

def _gen_2d_wh_round_fig(rcount, ccount, *, center, radius, color):
    """
    Draw a round at 'center' with 'radius' and 'color'.
    Parameters:
        rcound/ccount: Number of rows/columns of the output figure.
        center: tuple indicating the center of the round. The origin if the left-top of imgage. It may contain
            corrdinates exceeding the image.
        radius: integer indicating the radius.
        color: 'white' or 'black' indicating the color of round.
    Returns:
        Image data.
    Notes: center + radius may cover nothing of the result imag. E.g., the (rcount, ccount) is (10,10), and the 
    center is (50,50) and radius is 10, color is white. The result is a pure black rectangle.
    """
    r = np.arange(rcount)
    c = np.arange(ccount)
    i_idx, j_idx = np.meshgrid(r, c, indexing = 'ij')
    i_idx -= center[0]
    j_idx -= center[1]
    d = np.power(i_idx, 2) + np.power(j_idx, 2) - (radius ** 2)
    fg = 1
    bg = 2
    d[ d > 0 ] = bg
    d[ d <= 0] = fg
    fgc = 0 if 'black' == color else 255
    bgc = 255 - fgc
    d[fg == d] = fgc
    d[bg == d] = bgc
    return d

__formation_collection = {
        'sin': _gen_2d_wh_sin_fig,
        'cos': _gen_2d_wh_cos_fig,
        'grid': _gen_2d_wh_grid_fig,
        'round': _gen_2d_wh_round_fig,
                }

__gen_2d_wh_pattern_fig_help_str = \
"""
Generate a 2D figure of specified formation/pattern, with the color of white and black.
Parameters:
    formation: The formation/pattern of output figure. See below "Valid functions" for the valid formations.
    rcount/ccount: Number of row and column of the output figure.
    kwarg: Depending on the formation. See below.
Returns:
    A 2D ndarray.
"""
def gen_2d_wh_pattern_fig(formation, rcount, ccount, **kwarg):
    valid_formation = ', '.join(list(__formation_collection.keys()))
    if not (formation in __formation_collection): 
        print('valid formation: {}'.format(valid_formation))
        return None

    func = __formation_collection[formation]
    return func(rcount, ccount, **kwarg)

def shape_pixel_distance_power(row, col, orig = None):
    """
    Get an array containing the power of distance from specified point.
    Parameters:
        row/col: number of row and col of image.
        orig: opitonal. A tuple containing the point from which the distance is measured. If not specified (None),
        (ceil(row/2), ceil(col/2)), i.e. the center of image, is used.
    Returns:
        An np.ndarray containing the power of distances.
    """
    r_idxs, c_idxs = np.meshgrid(np.arange(row), np.arange(col), indexing = 'ij')
    if(None == orig):
        center = (int(np.ceil(row/2)), int(np.ceil(col/2)))
    else:
        center = orig
    r_idxs -= center[0]
    c_idxs -= center[1]
    return np.power(r_idxs, 2) + np.power(c_idxs, 2)

FREQ_FILT_LOW_PASS = 'lowpass'
FREQ_FILT_HIGH_PASS = 'highpass'
FREQ_FILTER_ILPF = 'ilpf'
FREQ_FILTER_IHPF = 'ihpf'
FREQ_FILTER_BLPF = 'blpf'
FREQ_FILTER_BHPF = 'bhpf'
FREQ_FILTER_GLPF = 'glpf'
FREQ_FILTER_GHPF = 'ghpf'
def _gen_ideal_freq_filter(row, col, *, center = None, D0, mode):
    """
    Generate an ideal frequnency filter.
    Parameters:
        row/col: size of the filter.
        center: center point of the filter.  If not specified (None), (ceil(row/2), ceil(row/2)) is used.
        D0: cutoff frequency.
        mode: FREQ_FILT_LOW_PASS or FREQ_FILT_HIGH_PASS.
    Returns:
        An np.ndarray containg the ideal frequency filter, valued in range [0, 1].

    Note:
        LPF: 1 if D <= D0 else 0
        HPF: 1 - LPF
        where D is the distance between point(u,v) to center.
    """
    d_power = shape_pixel_distance_power(row, col, center)
    D0_p = np.power(D0, 2)
    d_power[d_power <= D0_p] = 1
    d_power[d_power > D0_p] = 0
    if FREQ_FILT_LOW_PASS == mode:
        return d_power
    else:
        return 1 - d_power

def _gen_butterworth_freq_filter(row, col, *, center = None, D0, n = 2, mode):
    """
    Generate an butterworth frequnency filter.
    Parameters:
        row/col: size of the filter.
        center: center point of the filter.  If not specified (None), (ceil(row/2), ceil(row/2)) is used.
        D0: cutoff frequency.
        n: order, 2 as default.
        mode: FREQ_FILT_LOW_PASS or FREQ_FILT_HIGH_PASS.
    Returns:
        An np.ndarray containg the butterworth frequency filter, valued in range [0, 1].
    Note:
        LPF: H(u,v) = 1/(1 + power(D, 2*n)/power(D0, 2*n))
        HPF: 1 - LPF
        where D is the distance between point(u,v) to center.
    """
    d_power = shape_pixel_distance_power(row, col, center)
    d_power = np.power(d_power, n)
    D0_p = np.power(D0, 2 * n)
    H_uv = 1 / (1 + d_power/D0_p)
    if FREQ_FILT_LOW_PASS == mode:
        return H_uv
    else:
        return 1 - H_uv

def _gen_gaussian_freq_filter(row, col, *, center = None, D0, mode):
    """
    Generate an Gaussian frequnency filter.
    Parameters:
        row/col: size of the filter.
        center: center point of the filter.  If not specified (None), (ceil(row/2), ceil(row/2)) is used.
        D0: cutoff frequency.
        mode: FREQ_FILT_LOW_PASS or FREQ_FILT_HIGH_PASS.
    Returns:
        An np.ndarray containg the Gaussian frequency filter, valued in range [0, 1].
    Note:
        LPF: H(u,v) = power(e, -1 * power(D,2)/(2 * power(D0, 2))
        HPF: 1 - LPF
        where D is the distance between point(u,v) to center.
    """
    d_power = shape_pixel_distance_power(row, col, center)
    D0_p = np.power(D0, 2 * n)
    H_uv = np.exp(-1 * d_power / (2 * D0_p))
    if FREQ_FILT_LOW_PASS == mode:
        return H_uv
    else:
        return 1 - H_uv

__freq_filter_collection = {
        FREQ_FILTER_ILPF: lambda row, col, **kwarg: \
                _gen_ideal_freq_filter(row, col, mode = FREQ_FILT_LOW_PASS, **kwarg),
        FREQ_FILTER_IHPF: lambda row, col, **kwarg: \
                _gen_ideal_freq_filter(row, col, mode = FREQ_FILT_HIGH_PASS, **kwarg),
        FREQ_FILTER_BLPF: lambda row, col, **kwarg: \
                _gen_butterworth_freq_filter(row, col, mode = FREQ_FILT_LOW_PASS, **kwarg),
        FREQ_FILTER_BHPF: lambda row, col, **kwarg: \
                _gen_butterworth_freq_filter(row, col, mode = FREQ_FILT_HIGH_PASS, **kwarg),
        FREQ_FILTER_GLPF: lambda row, col, **kwarg: \
                _gen_gaussian_freq_filter(row, col, mode = FREQ_FILT_LOW_PASS, **kwarg),
        FREQ_FILTER_GHPF: lambda row, col, **kwarg: \
                _gen_gaussian_freq_filter(row, col, mode = FREQ_FILT_HIGH_PASS, **kwarg),
        }

__freq_filter_help_strs = {
        FREQ_FILTER_ILPF: _gen_ideal_freq_filter.__doc__,
        FREQ_FILTER_IHPF: _gen_ideal_freq_filter.__doc__,
        FREQ_FILTER_BLPF: _gen_butterworth_freq_filter.__doc__,
        FREQ_FILTER_BHPF: _gen_butterworth_freq_filter.__doc__,
        FREQ_FILTER_GLPF: _gen_gaussian_freq_filter.__doc__,
        FREQ_FILTER_GHPF: _gen_gaussian_freq_filter.__doc__,
        }

__gen_freq_filter_help_str = \
"""
Generate frequency domain filter.
Paramters:
    fn: filter name. See below for valid values.
    row/col: size of the filter.
    kwarg: parameters depending on filter type.
Returns:
    An np.ndarray containing the filter, valued in range [0, 1].
Note:
    'fn' contains the 'low pass' or 'high pass' information, so user of this function needs not pass the 'mode'
    parameter if it is listed in the detailed help string of filter.
"""
def gen_freq_filter(fn, row, col, **karg):
    return __freq_filter_collection[fn](row, col, **kwarg) 

__INIT_HELP_STR_S = 'init_help_s'
__WRAP_DICT_S = 'wrap_dict'
__FUNC_DOC_S = 'func_doc'
__wrap_functions_help_collection = [
        {
            __FUNC_S: gen_2d_wh_pattern_fig,
            __INIT_HELP_STR_S: __gen_2d_wh_pattern_fig_help_str,
            __WRAP_DICT_S: __formation_collection,
            __FUNC_DOC_S: None,
            },
        {
            __FUNC_S: gen_freq_filter,
            __INIT_HELP_STR_S: __gen_freq_filter_help_str, 
            __WRAP_DICT_S: __freq_filter_collection,
            __FUNC_DOC_S: __freq_filter_help_strs,
            },
       ]
def _construct_help_str_for_wrap_func():
    for item in __wrap_functions_help_collection:
        init_str = item[__INIT_HELP_STR_S]
        wrap_dics = item[__WRAP_DICT_S]
        func_docs = item[__FUNC_DOC_S]
        valid_functions = ', '.join(list(wrap_dics.keys()))
        valid_functions += '.'
        seperator_line = '-' * 10
        help_str = '{}\n{}\nValid functions: {}\n{}\n'.\
                format(init_str, seperator_line, valid_functions, seperator_line)
        for func in wrap_dics:
            if func_docs != None:
                f_str = func_docs[func]
            else:
                f_str = wrap_dics[func].__doc__
            help_str += '{}\n{}\n{}\n'.format(func, f_str, seperator_line)
        item[__FUNC_S].__doc__ = help_str

_construct_help_str_for_wrap_func()

