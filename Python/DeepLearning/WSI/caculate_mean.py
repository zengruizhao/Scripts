# coding=utf-8
'''
    caculate the mean of data
    author: Zengrui Zhao
'''
import numpy as np
import os
from skimage import io
from scipy import misc
import cv2
import sys

def calculateWithoutSubFolder(path):
    temp = np.zeros((224, 224, 3))
    img_number = 0
    for img in os.listdir(path):
        print img
        data = io.imread(os.path.join(path, img))
        temp += data
        img_number += 1
    mean = temp / img_number
    return mean


def calculateWithSubFolder(path):
    temp = np.zeros((224, 224, 3))
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
    mean = calculateWithSubFolder(path = '/home/zzr/Data/Skin/data/All_2')
    # mean = calculateWithoutSubFolder(path = '/home/zzr/Data/ISIC/densenet/val'):

    np.save('/home/zzr/Data/Skin/script_all/mean.npy', mean)