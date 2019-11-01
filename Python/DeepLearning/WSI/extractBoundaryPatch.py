# coding=utf-8
from semanticXml import SemanticXml
from get_small_patch import read_image, find_roi_bbox_1
import os
import numpy as np
import matplotlib.pyplot as plt
img_WSI_dir='/home/zzr/Desktop'
name = '2019-02-28 15.44.02.ndpi'
thumbnail_mask = np.load('/media/zzr/Data/skin_xml/phase2/mask/lowlevel.npy')
semanticxml = SemanticXml(thumbnail_mask, 0, img_WSI_dir, name)
slide, downsample_image, level, m, n = read_image(os.path.join(img_WSI_dir, name), 36)
bounding_boxes, rgb_contour, image_dilation = find_roi_bbox_1(downsample_image, show=False)
print image_dilation.shape
image = semanticxml.semantic2xml(resultFile=None, seg_path='/media/zzr/Data/skin_xml/phase2/semantic/result')
print image.shape
# plt.imshow(image)
# plt.show()