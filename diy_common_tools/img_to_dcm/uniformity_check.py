import numpy as np
import cv2 as cv
import sys

usage_str = "usage: uniformity_check img_file"
if len(sys.argv) < 2:
    print(usage_str)
    sys.exit(-1)

sample_area_size = (64, 64)
sample_area_o_coe = ((1/6, 1/2, 5/6), (1/6, 1/2, 5/6))

def get_sample_areas(img_w, img_h):
    """return a list, containning the (x,y,w,h) of every sample area"""
    centers = [(int(img_w * x_coe), int(img_h * y_coe)) 
            for x_coe in sample_area_o_coe[0] for y_coe in sample_area_o_coe[1]]
    areas = []
    for c in centers:
        x = int(c[0] - sample_area_size[0]/2)
        if x < 0 or x >= img_w: x = 0
        y = int(c[1] - sample_area_size[1]/2)
        if y < 0 or y >= img_h: y = 0
        w = sample_area_size[0]
        if x + w > img_w: w = img_w - x
        h = sample_area_size[1]
        if y + h > img_h: h = img_h - y
        areas.append((x,y,w,h))
    return areas

img_f = sys.argv[1]
img = cv.imread(img_f, cv.IMREAD_UNCHANGED)
if len(img.shape) > 3:
    print("image file has " + str(len(img.shape)) + " channels, but we can only process image of 2 or file channels.")
    sys.exit(-2)

if len(img.shape) == 3:
    img = img[:,:,0]

s_areas = get_sample_areas(img.shape[1], img.shape[0])
a_num = len(s_areas)
vis = tuple(np.average(img[a[0]:a[0]+a[2], a[1]:a[1]+a[3]]) for a in s_areas)
for i in range(a_num): print(s_areas[i], end=""); print(":\t" + str(vis[i]))

vm = np.average(vis)
R = (sum(((vi - vm)**2 for vi in vis))/(a_num - 1 if a_num > 1 else 1))**(1/2)
print("R is: " + str(R))
print("Vm is: " + str(vm))
print("R/Vm is: " + str(R/vm))
