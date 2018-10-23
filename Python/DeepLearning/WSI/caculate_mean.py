# coding=utf-8
"""
    caculate the mean of data
    author: Zengrui Zhao
"""
import numpy as np
import os
from skimage import io
from scipy import misc
import cv2
import sys


def calculateWithoutSubFolder(path):
    temp = np.zeros((144, 144, 3))
    img_number = 0
    for img in os.listdir(path):
        print img
        data = io.imread(os.path.join(path, img))
        temp += data
        img_number += 1
    mean = temp / img_number
    return mean


def calculateWithSubFolder(path):
    temp = np.zeros((144, 144, 3))
    img_number = 0
    for sub in os.listdir(path):
        for img in os.listdir(os.path.join(path, sub)):
            # img = unicode(img, 'utf-8')
            print img
            data = io.imread(os.path.join(os.path.join(path, sub), img)).astype(np.float32)
            # data = misc.imread(os.path.join(path, img).decode('utf-8').encode('gbk'))
            # data = cv2.imdecode(np.fromfile(os.path.join(path, img), dtype=np.uint8), -1)
            # data = cv2.imread(os.path.join(path, img).encode('gbk'))
            temp += data
            img_number += 1
    mean = temp / img_number
    return mean


if __name__ == '__main__':
    # mean = calculateWithSubFolder(path='/media/zzr/SW/Skin_xml/WSI_20/WSI/Train_test/train')
    mean = calculateWithoutSubFolder(path='/media/zzr/Data/skin_xml/original_new/2018-08-09 153008')

    np.save('/media/zzr/Data/skin_xml/original_new/2018-08-09 153008.npy', mean)