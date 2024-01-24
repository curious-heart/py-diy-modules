import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import cv2 as cv
import sys
import os

def get_data_from_raw(raw_image_path, i_shape, data_type):
    """
    i_shape: (width, height)
    data_type: np.uint16 or np.uint8
    """
    img = np.fromfile(raw_image_path, dtype = data_type)
    img = np.reshape(img, tuple(reversed(i_shape)))
    return img

def save_to_dcm(pth_with_bn, img, px_bits):
    print("Setting file meta information...")
    filename = pth_with_bn + '.dcm' 
    # Populate required values for file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    print("Setting dataset values...")
    # Create the FileDataset instance (initially no data elements, but file_meta
    # supplied)
    ds = FileDataset(filename, {},
                      file_meta=file_meta, preamble=b"\0" * 128)
                
    # # Write as a different transfer syntax XXX shouldn't need this but pydicom
    # # 0.9.5 bug not recognizing transfer syntax
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    ds.Modality = "DX"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ds.BitsAllocated = px_bits
    ds.BitsStored = ds.BitsAllocated
    ds.HighBit = ds.BitsStored - 1
    ds.SamplesPerPixel = 1
    ds.PlanarConfiguration = 0

    ds.InstanceNumber = 1
    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"

    ds.ImagesInAcquisition = "1"
    ds.Rows, ds.Columns = img.shape[:2]
    ds.PixelSpacing = [0.139, 0.139]
    ds.PixelRepresentation = 0
    ds.PixelData= img.tobytes()
    # import pdb;pdb.set_trace()
    ds.PhotometricInterpretation = 'MONOCHROME2'
    
    print("Writing file", filename)
    ds.save_as(filename)
    print("File saved.")

def str2tuple(s):
    """
    input: a string like '(3,25,44)'
    return: a tuple (3,25,44)
    """
    t = s.replace(' ', '')[1:-1].split(',')
    return tuple(int(i) for i in t)

_usage_str = \
"""
用法1:\n\traw2dcm.py raw文件名 图像形状 位深
\t- raw文件名后缀为.raw或.aof或空
\t- 图像形状格式为：(宽度,高度)；位深为8或16
输出tiff文件和dcm文件

用法2:\n\traw2dcm.py 图像文件名(如.tiff或.png文件)
输出raw文件和dcm文件

注意：
1) 对于多channel的图像文件，仅处理第一个channel
2) 只能处理16位或8位的数据
"""

if len(sys.argv) < 2:
    print(_usage_str)
    sys.exit(0)

input_file_fpn = sys.argv[1]
fbn, ext = os.path.splitext(input_file_fpn)
if (not ext) or ('.raw' == ext) or ('.aof' == ext):
    #raw file
    if len(sys.argv) < 4:
        print(_usage_str)
        sys.exit(0)
    img_shape_str, dept = sys.argv[2], int(sys.argv[3])
    img_shape = str2tuple(img_shape_str)
    data_type = np.uint16 if 16 == dept else np.uint8
    pixel_bits = 16 if np.uint16 == data_type else 8
    image_data = get_data_from_raw(input_file_fpn, img_shape, data_type)
    cv.imwrite(fbn + ".tif", image_data, (cv.IMWRITE_TIFF_COMPRESSION, 1))
else:
    #image file
    image_data = cv.imread(input_file_fpn, cv.IMREAD_UNCHANGED)
    if len(image_data.shape) > 3:
        print(_usage_str); sys.exit(0)
    elif len(image_data.shape) == 3:
        image_data = image_data[:,:,0]
    image_data.tofile(fbn + ".raw")
    img_shape = tuple(reversed(image_data.shape[0:2]))
    data_type = image_data.dtype
    pixel_bits = 16 if np.uint16 == data_type else 8

save_to_dcm(fbn, image_data, pixel_bits)
