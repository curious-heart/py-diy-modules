import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import cv2 as cv
import sys
import os

def save_to_dcm(raw_image_path, i_shape):
    pth_with_bn, ext_n = os.path.splitext(raw_image_path)
    img = np.fromfile(raw_image_path, dtype=np.uint16)
    img = np.reshape(img, tuple(reversed(i_shape)))
    cv.imwrite(pth_with_bn + ".tif", img, (cv.IMWRITE_TIFF_COMPRESSION, 1))

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

    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
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

if len(sys.argv) < 3:
    print("用法:\n\traw2dcm.py raw文件名 图像形状")
    print("\t图像形状格式为：(宽度,高度)")
    sys.exit(0)

def str2tuple(s):
    """
    input: a string like '(3,25,44)'
    return: a tuple (3,25,44)
    """
    t = s.replace(' ', '')[1:-1].split(',')
    return tuple(int(i) for i in t)

#save_to_dcm("original.aof", (3072, 3072))
raw_file_fpn, img_shape = sys.argv[1:]
img_shape = str2tuple(img_shape)
save_to_dcm(raw_file_fpn, img_shape)
