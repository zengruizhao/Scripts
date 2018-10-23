# coding=utf-8
"""
    segment whole slide img with specific stride after preprocessing the slide
    Tricks: bigger batch_size and less print information, faster speed
    author: ccf
    modified by: Zengrui Zhao 2018.7.19
"""


import sys
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import os
import skimage
from skimage import io
import warnings
import matplotlib.pyplot as plt
import cv2

warnings.filterwarnings('ignore')
##
caffe_root = '/home/zzr/caffe/'
sys.path.insert(0, caffe_root + 'python')
caffe.set_mode_gpu()
caffe.set_device(0)

All_start = time.time()
deploy = '/home/zzr/Data/Skin/script_all/deploy.prototxt'
mean = '/home/zzr/Data/Skin/script_all/mean.npy'
model = '/home/zzr/Data/Skin/script_all/model/_iter_10000.caffemodel'
img_WSI_dir = '/run/user/1000/gvfs/smb-share:server=darwin-mi,share=data/Skin/MF_10/20180627/'
save_file = '/media/zzr/Data/skin_xml/mask_result/'

mean_data = np.transpose(np.load(mean), (2, 0, 1))
# =====================================
net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_mean('data', mean_data.mean(1).mean(1))
transformer.set_raw_scale('data', 255)  # 像素放大到255
transformer.set_channel_swap('data', (2, 1, 0))  # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
# net.blobs['data'].reshape(200, 3, 224, 224)  # 将输入图片格式转化为合适格式（与deploy文件相同）
batchsize = net.blobs['data'].shape[0]
# =====================================


def get_bbox(cont_img, rgb_image=None):
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour


def find_roi_bbox(rgb_image):
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([7, 7, 7])
    upper_red = np.array([250, 250, 250])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red)  # lower20===>0,upper200==>0

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    image_dilation = cv2.morphologyEx(np.array(image_open), cv2.MORPH_DILATE, open_kernel)
    bounding_boxes, rgb_contour = get_bbox(image_dilation, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_dilation


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[- 1].split('.ndpi')[0]
    return filename


def read_image(image_path, stride):
    try:
        image = OpenSlide(image_path)
        w, h = image.dimensions
        n = int(math.floor((h - 0) / stride))
        m = int(math.floor((w - 0) / stride))
        level = image.level_count - 1
        downsample_image = np.array(image.read_region((0, 0), level, image.level_dimensions[level]))[..., 0:-1]
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None, None

    return image, downsample_image, level, m, n


labels = ['背景', '表皮', '真皮', '脂肪', '毛囊', '汗腺']
img_WSI_all = []
img_WSI_all.append(os.path.join(img_WSI_dir, '2018-06-06 15.14.49.ndpi'))
# train:2018-06-06 15.14.49.ndpi 2018-06-06 16.14.09.ndpi#test:2018-06-06 15.15.56.ndpi
for name1 in img_WSI_all:
    start = time.time()
    stride = 56 # set stride
    name = get_filename_from_path(name1)
    slide, downsample_image, level, m, n = read_image(name1, stride)
    bounding_boxes, rgb_contour, image_dilation = find_roi_bbox(downsample_image)
    mask = np.zeros([n, m])
    print '{} rows, {} columns'.format(n, m)
    print('%s Classification is in progress' % name1)
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2]+1)
        b_y_end = int(bounding_box[1]) + int(bounding_box[3]+1)
        mag_factor = 2 ** level
        col_cords = np.arange(b_x_start*mag_factor/stride, b_x_end*mag_factor/stride)
        row_cords = np.arange(b_y_start*mag_factor/stride, b_y_end*mag_factor/stride)
        ii = []
        jj = []
        temp = 0
        for i in col_cords:
            for j in row_cords:
                if int(image_dilation[stride * j // mag_factor, stride * i // mag_factor]) != 0:
                    ii.append(i)  # the location of i, j
                    jj.append(j)
                    img_tmp = skimage.img_as_float(
                        np.array(slide.read_region((stride * i, stride * j),
                                                   0, (224, 224)))).astype(np.float32)[..., 0:-1]
                    net.blobs['data'].data[temp] = transformer.preprocess('data', img_tmp)
                    temp += 1

                    if temp == batchsize or (i == col_cords[-1] and j == row_cords[-1]):
                        net.forward()
                        for idx in range(batchsize):
                            prob = net.blobs['prob'].data[idx].flatten()
                            order = prob.argmax()
                            # print '%d / %d, idx, the class is %s %s' % (i, col_cords[-1], labels[order], np.max(prob))
                            mask[jj[idx], ii[idx]] = order
                        ii = []
                        jj = []
                        temp = 0
                        print '%d / %d' % (i, col_cords[-1])

    np.save(save_file + 'Result_' + name + '.npy', mask)
    print('has done...')
All_end = time.time()
print(All_end - All_start)
print('success')