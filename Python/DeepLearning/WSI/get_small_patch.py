# coding=utf-8
import cv2
import numpy as np
import os
from multiprocessing import Process
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import skimage
from skimage import io
import warnings
warnings.filterwarnings('ignore')

img_WSI_dir = '/media/zzr/Data/skin_xml'
fore_path = '/media/zzr/Data/skin_xml/original/fore/'
back_path = '/media/zzr/Data/skin_xml/original/back/'


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


def get_patch(startCol, endCol, row_cords, stride, image_dilation,
              mag_factor, slide):
    col_cords = np.arange(startCol, endCol)
    for i in col_cords:
        print i, '/', col_cords[-1]
        for j in row_cords:
            if int(image_dilation[stride * j // mag_factor, stride * i // mag_factor]) != 0:
                img = skimage.img_as_float(
                    np.array(slide.read_region((stride * i, stride * j),
                                               0, (224, 224)))).astype(np.float32)[..., 0:-1]
                save_path = fore_path  # if np.mean(img) < 0.9 else back_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                name = str(stride * i) + '_' + str(stride * j) + '.jpg'
                # print np.mean(img)
                io.imsave(os.path.join(save_path, name), img)


def main():
    start = time.time()
    nProcs = 8
    ProcessPointer = [None] * nProcs
    img_WSI_all = []
    img_WSI_all.append(os.path.join(img_WSI_dir, '2018-06-06 15.14.49.ndpi'))
    # train:2018-06-06 15.14.49.ndpi 2018-06-06 16.14.09.ndpi#test:2018-06-06 15.15.56.ndpi
    for name1 in img_WSI_all:
        stride = 56  # set stride
        slide, downsample_image, level, m, n = read_image(name1, stride)
        bounding_boxes, rgb_contour, image_dilation = find_roi_bbox(downsample_image)
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
            ColPerCore = len(col_cords) / nProcs
            for proc in xrange(nProcs):
                startCol = col_cords[0] + ColPerCore * proc
                endCol = min(ColPerCore * (proc + 1), len(col_cords))
                ProcessPointer[proc] = Process(target=get_patch, args=(
                        startCol, endCol, row_cords, stride, image_dilation,
                        mag_factor, slide))
                ProcessPointer[proc].start()

            for proc in xrange(nProcs):
                ProcessPointer[proc].join()

    print 'Time:', time.time() - start


if __name__ == '__main__':
    main()