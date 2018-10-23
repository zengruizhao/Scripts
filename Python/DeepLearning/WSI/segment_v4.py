# coding=utf-8

#author:caichengfei

caffe_root = '/home/ccf/CaffeMex_densenet_focal_loss/'
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
from skimage import measure,morphology
import matplotlib.pyplot as plt
import scipy
from scipy import misc

caffe.set_mode_gpu()
caffe.set_device(0)

All_start = time.time()
deploy = '/home/ccf/CCF/Colorecal-cancer/SMU_data_v4_densenet/prototxt/DenseNet_201_focal_loss_deploy.prototxt'
mean = '/home/ccf/CCF/Colorecal-cancer/SMU_data_V3/code/mean.npy'
model = '/home/ccf/CCF/Colorecal-cancer/SMU_data_v4_densenet/model_densnet_focal_loss/caffe_colorecal_DenseNet_201_iter_100000.caffemodel'
img_WSI_dir = '/home/ccf_disk/data/Colorecal_cancer_South_Hospital/survival_experiment_data/284/'
label_filename = '/home/ccf/CCF/Colorecal-cancer/SMU_data_V3/code/labels.txt'
save_file='/home/ccf/CCF/Colorecal-cancer/SMU_data_v4_densenet/densnet_focal_loss_image_survival_result_npy/'
patch_save_path = '/home/ccf_disk/Colorecal-cancer/SMU_data_v4_densenet/'
#=====================================
net = caffe.Net(deploy, model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))  # python读取的图片文件格式为H×W×K，需转化为K×H×W
transformer.set_mean('data', np.load(mean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)  # python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
transformer.set_channel_swap('data', (2, 1, 0))  # caffe中图片是BGR格式，而原始格式是RGB，所以要转化
net.blobs['data'].reshape(50, 3, 150, 150)  # 将输入图片格式转化为合适格式（与deploy文件相同）
batchsize = net.blobs['data'].shape[0]
# print batchsize
#=====================================

def get_imlist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ndpi')]

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename

def read_image(image_path):
    try:
        image = OpenSlide(image_path)
        w, h = image.dimensions
        # w, h = image.level_dimensions[1]
        n = int(math.floor((h - 0) / 150))
        m = int(math.floor((w - 0) / 150))
        level = image.level_count - 1
        downsample_image = np.array(image.read_region((0, 0), level, image.level_dimensions[level]))
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return image, downsample_image, level, m, n

def mkdir(path,name):
    patch_path = (path + name[0:9] + '/')
    isExist = os.path.exists(patch_path)
    if not isExist:
        os.makedirs(patch_path)
        print('folder name :', patch_path)
    else:
        print('folder already exists..... ')

    return patch_path

img_WSI_all = get_imlist(img_WSI_dir)
for name1 in img_WSI_all:
    start = time.time()
    name = get_filename_from_path(name1)
    slide, downsample_image, level, m, n = read_image(name1)
    mask = np.zeros([n,m])
    print('%s Classification is in progress' % name1)
    patch_path = mkdir(patch_save_path, name)

    for i in range(1, m):
        nbatch = (n+batchsize-1)//batchsize
        for k in range(nbatch):
            idx = np.arange(k*batchsize, min(n, (k+1)*batchsize))
            for tdx in idx:
                indexofdata = tdx%batchsize
                img_tmp = skimage.img_as_float(np.array(slide.read_region((0+150*(i-1), 0+150*(tdx)), 1, (150, 150)))).astype(np.float32)
                # img_tmp0 = skimage.img_as_float(np.array(slide.read_region((0+150*(i-1), 0+150*(tdx)), 0, (150, 150)))).astype(np.float32)
                net.blobs['data'].data[indexofdata] = transformer.preprocess('data',img_tmp)
                out = net.forward()
                labels = np.loadtxt(label_filename, str, delimiter='\t')
                prob = net.blobs['prob'].data[indexofdata].flatten()
                # print prob
                prob1= out['prob'][indexofdata][0]
                order=prob.argsort()[9]
                if order == 3:
                    Path = patch_path + name[0:9] + '_' + str(i) + '_' + str(tdx) + '.png'
                    scipy.misc.imsave(Path, img_tmp)
                # print'the class is',labels[order]
                # a = np.ones((10,10))
                # b = prob*np.array(a)
                b=np.full((1, 1), order)
                mask[tdx:(tdx+1), (i-1):i] = b
                # print i,tdx
    end = time.time()
    np.save(save_file+'Result_'+name+'_100000.npy', mask)
    print('has done...')
    print(end-start)
All_end = time.time()
print(All_end-All_start)
print('success')