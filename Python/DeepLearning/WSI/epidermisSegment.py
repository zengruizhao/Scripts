# coding=utf-8
from get_small_patch import read_image
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, try_all_threshold, threshold_mean
import numpy as np

img_WSI_dir = '/media/zzr/Data/skin_xml/original_new/RawImage/'
stride = 36
WSI = ['2018-08-09 15.33.04.ndpi']


def segment_epidermis(rgb_image):
    rgb_image = 255-rgb_image
    thres = threshold_otsu(rgb_image[..., 0])
    mask = (rgb_image[..., 0] > thres).astype('uint8')
    plt.imshow(mask)
    plt.show()


for name1 in WSI:
    slide, downsample_image, level, m, n = read_image(os.path.join(img_WSI_dir, name1), stride)
    segment_epidermis(downsample_image)
