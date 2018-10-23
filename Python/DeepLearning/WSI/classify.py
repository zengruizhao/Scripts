# coding=utf-8

#author:caichengfei

caffe_root = '/home/zzr/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import cv2
import os
import skimage
from skimage import io
import csv
from skimage import measure,morphology
import matplotlib.pyplot as plt
import scipy
from scipy import misc

caffe.set_mode_gpu()
caffe.set_device(0)

All_start = time.time()
deploy = '/home/zzr/caffe/models/SENET/ISIC/SE_ResNet_50/deploy.prototxt'
mean = '/home/zzr/SegNet/Scripts/ccf/mean.npy'
model = '/home/zzr/caffe/models/SENET/ISIC/SE_ResNet_50/model/_iter_60000.caffemodel'
img_path = '/home/zzr/Data/ISIC/densenet/test'
mean_data = np.transpose(np.load(mean), (2, 0, 1))
#=====================================
net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_mean('data', mean_data.mean(1).mean(1))
transformer.set_raw_scale('data', 255)  # python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
transformer.set_channel_swap('data', (2, 1, 0))  # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
net.blobs['data'].reshape(1, 3, 224, 224)  # 将输入图片格式转化为合适格式（与deploy文件相同）
batchsize = net.blobs['data'].shape[0]
# print batchsize
#=====================================
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

def get_imlist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


img_all = get_imlist(img_path)
result = [['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']]
for name1 in img_all:
    temp = []
    start = time.time()
    img_tmp = caffe.io.load_image(name1)
    net.blobs['data'].data[0] = transformer.preprocess('data', img_tmp)
    net.forward()
    prob = net.blobs['prob'].data[0].flatten()
    # index = np.argmax(prob)
    # prob[0:7] = 0.0
    # prob[index] = 1.0
    temp.append(name1.split('/')[-1].split('.')[0])
    temp[1:8] = prob[0:7]
    end = time.time()
    result.append(temp)
    print 'time:', end-start
with open('test.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result)

print('success')