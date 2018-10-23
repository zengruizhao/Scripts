# coding=utf-8
"""
    segment whole slide img according to saved img
    author: Zengrui Zhao 2018.8.30
"""

caffe_root = '/home/zzr/caffe/'
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import time
import os
from skimage import io
import warnings
import math
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


All_start = time.time()
deploy = '/home/zzr/Data/Skin/script_all/deploy.prototxt'
mean = '/home/zzr/Data/Skin/script_all/mean.npy'
model = '/home/zzr/Data/Skin/script_all/model/_iter_10000.caffemodel'
save_file = '/media/zzr/Data/skin_xml/mask_result/'
labels = ['背景', '表皮', '真皮', '脂肪', '毛囊', '汗腺']


def caffe_init():
    caffe.set_mode_gpu()
    caffe.set_device(0)

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
    return transformer, net, batchsize


def get_xy(img_file):
    min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
    for img in img_file:
        x_ = int(img.split('.')[0].split('_')[0])
        y_ = int(img.split('.')[0].split('_')[1])
        min_x = x_ if x_ < min_x else min_x
        min_y = y_ if y_ < min_y else min_y
        max_x = x_ if x_ > max_x else max_x
        max_y = y_ if y_ > max_y else max_y

    return min_x, min_y, max_x, max_y


def main():
    start = time.time()
    transformer, net, batchsize = caffe_init()
    img_path = '/media/zzr/Data/skin_xml/original/fore'
    img_file = os.listdir(img_path)
    stride = 56
    min_x, min_y, max_x, max_y = get_xy(img_file)
    mask = np.zeros(np.array([max_y, max_x]) / stride + 1)
    epoch = math.ceil(len(img_file) / batchsize) + 1
    x, y = [], []
    temp = 0
    for idx in xrange(int(epoch)):
        print 'epoch', idx, '/', epoch
        img_file_epoch = img_file[(idx * batchsize):min((idx+1) * batchsize, len(img_file))]
        temp += len(img_file_epoch)
        for index, img in enumerate(img_file_epoch):
            file = io.imread(os.path.join(img_path, img)).astype('float')/255
            net.blobs['data'].data[index] = transformer.preprocess('data', file)
            x.append(int(img.split('.')[0].split('_')[0]))
            y.append(int(img.split('.')[0].split('_')[1]))

        net.forward()
        for index, _ in enumerate(img_file_epoch):
            prob = net.blobs['prob'].data[index].flatten()
            order = prob.argmax()
            mask[y[index] / stride, x[index] / stride] = order
            print str(y[index]), '_', str(x[index]), 'the class is:', labels[order], np.max(prob)

        x, y = [], []

    np.save(save_file + 'Result.npy', mask)
    print time.time() - start


if __name__ == '__main__':
    main()