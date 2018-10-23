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


txtpath='/home/zzr/Data/contract_exp_data/NKI/1_1/test.txt'

caffe_root = '/home/zzr/caffe-segnet-cudnn5/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


IMAGE='/home/zzr/Data/contract_exp_data/NKI/1_1/test/img/'
#IMAGE_GT='/home/zzr/Data/temp/train_mask/'
IMAGE_seg='/home/zzr/Data/contract_exp_data/NKI/1_1/script/result/'
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

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	##
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()

	BG = [0,0,0]
	M = [0,255,0]
	label_colours = np.array([BG, M])
	for l in range(0,2):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]

	#scipy.misc.imsave(IMAGE + name +'.jpg',image)
	#scipy.misc.imsave(IMAGE_GT +name +'_gt.jpg',rgb_gt)
	scipy.misc.imsave(IMAGE_seg + name +'_seg.png',g) #rgb
	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')

	#plt.figure()
	#plt.imshow(image,vmin=0, vmax=1)
	#plt.figure()
	#plt.imshow(rgb_gt,vmin=0, vmax=1)
	#plt.figure()
	#plt.imshow(rgb,vmin=0, vmax=1)
	#plt.show()


print 'Success!'

