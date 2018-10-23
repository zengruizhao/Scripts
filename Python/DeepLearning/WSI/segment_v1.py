# coding=utf-8

#author:caichengfei

caffe_root = '/home/ccf/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import openslide
import time
import math
import cv2
import skimage
from skimage import measure,morphology
import matplotlib.pyplot as plt
# from scipy import misc

caffe.set_mode_gpu()
caffe.set_device(0)

start=time.time()

deploy = '/home/ccf/CCF/Colorecal-cancer/SUM_data_V1/profile/prototxt/deploy.prototxt'
mean = '/home/ccf/CCF/Colorecal-cancer/SUM_data_V1/profile/prototxt/mean.npy'
model = '/home/ccf/CCF/Colorecal-cancer/SUM_data_V1/profile/model/SUM_150_Colorecal-cancer__iter_60000.caffemodel'
#img_WSI = '/home/ccf/CCF/creat_data/test/data/ndpi2/2017-01-14_23_05_30.ndpi'
img_WSI='/home/ccf/CCF/creat_data/test/data/leave-image/2017-01-14-20-03-58.ndpi'
label_filename = '/home/ccf/CCF/Colorecal-cancer/SUM_data_V1/profile/prototxt/labels.txt'

slide=openslide.open_slide(img_WSI)
[w,h]=slide.dimensions
OVslide=slide.level_dimensions[5]#Overview
img=np.array(slide.read_region((0,0),5,OVslide))
GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#用cvtColor获得原图像的副本
blur = cv2.GaussianBlur(GrayImage,(5,5),0)#用高斯平滑处理原图像降噪,模糊图片
ret1,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th1=~th1

# Osmall1=np.zeros((2000,OVslide[0]))
# Osmall2=np.zeros((OVslide[1]-6500,OVslide[0]))#16====>1400
# th1[:2000,:OVslide[0]]=Osmall1
# th1[6500:OVslide[1],:OVslide[0]]=Osmall2
# plt.imshow(th1)
# plt.show()

dst=morphology.remove_small_objects(th1,min_size=1000,connectivity=1)
# plt.imshow(dst)
# plt.show()

bbox=measure.regionprops(dst)[0]['bbox']
bbox=32*np.array(bbox)
[x1,y1,x2,y2]=bbox
n=int(math.floor((x2-x1)/150))
m=int(math.floor((y2-y1)/150))
mask=np.zeros([10*n,10*m])

net = caffe.Net(deploy,model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))# python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)# python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
transformer.set_channel_swap('data', (2, 1, 0))# caffe中图片是BGR格式，而原始格式是RGB，所以要转化
net.blobs['data'].reshape(50,3,150,150)# 将输入图片格式转化为合适格式（与deploy文件相同）
batchsize = net.blobs['data'].shape[0]
for i in range(1,m):
    nbatch = (n+batchsize-1)//batchsize
    for k in range(nbatch):
        idx = np.arange(k*batchsize,min(n,(k+1)*batchsize))
        for tdx in idx:
            indexofdata = tdx%batchsize
            img_tmp=skimage.img_as_float(np.array(slide.read_region((y1+150*(i-1),x1+150*(tdx)),0,(150,150)))).astype(np.float32)
            # img1 = np.tile(img_tmp,(1,1,3))
            # img2 = img1[:,:,3]
            # plt.imshow(img_tmp)
            # plt.show()
            net.blobs['data'].data[indexofdata] = transformer.preprocess('data',img_tmp)
        out = net.forward()
        for tdx in idx:
            labels=np.loadtxt(label_filename,str,delimiter='\t')
            indexofdata = tdx%batchsize
            prob=net.blobs['prob'].data[indexofdata].flatten()
            # print prob
            prob1= out['prob'][indexofdata][0]
            order=prob.argsort()[7]
            print'the class is',labels[order]
            # a = np.ones((10,10))
            # b = prob*np.array(a)
            b=np.full((10,10),order)
            mask[10*tdx:10*(tdx+1),10*(i-1):10*i] = b
            print i,tdx
end = time.time()
np.save("/home/ccf/CCF/Colorecal-cancer/SUM_data_V1/result/npy/Result_2017-01-14-20-03-58_60000.npy",mask)
print('has done...')
print(end-start)
