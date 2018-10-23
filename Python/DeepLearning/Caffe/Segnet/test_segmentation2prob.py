# -*- coding: utf-8 -*-
#author:caichengfei

import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize

#open txt的模块
import linecache
import re

def getline(thefilepath,line_num):
    if line_num < 1 :return ''
    for currline,line in enumerate(open(thefilepath,'rU')):
        if currline == line_num -1 : return line
    return ''


txtpath='/home/zzr/Data/IDRiD/New/OD/Data/pre_aug/train.txt'

caffe_root = '/home/zzr/caffe-segnet-cudnn5/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


IMAGE='/home/zzr/Data/IDRiD/New/OD/Data/pre_aug/train_img/'
#IMAGE_GT='/home/zzr/Data/temp/train_mask/'
IMAGE_seg='/home/zzr/Data/IDRiD/New/OD/Data/seg_result/unet_train/'
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

    theline = linecache.getline(txtpath, i+1)
    name = theline.split(' ')
    text = name[0]
    m = re.split('/', text)
    name = m[len(m) - 1][0:len(m[len(m) - 1]) - 4]

    net.forward()

    predicted = net.blobs['prob'].data
    
    scipy.misc.imsave(IMAGE_seg + name +'_seg.png', predicted[0,1,...]) #rgb

print 'Success!'

