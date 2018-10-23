# coding=utf-8
'''
    segment whole slide img with specific stride
    author: ccf
    modified by: Zengrui Zhao 2018.7.19
'''

caffe_root = '/home/zzr/caffe/'
import sys

sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time
import math
import os
import skimage
from skimage import io
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


caffe.set_mode_gpu()
caffe.set_device(0)

All_start = time.time()
deploy = '/home/zzr/Data/Skin/script_all/deploy.prototxt'
mean = '/home/zzr/Data/Skin/script_all/mean.npy'
model = '/home/zzr/Data/Skin/script_all/model/_iter_10000.caffemodel'
img_WSI_dir = '/run/user/1000/gvfs/smb-share:server=darwin-mi.local,share=data/Skin/MF 皮研所 10张/20180627/'
save_file = '/media/zzr/Data/skin_xml/mask_result/'
patch_save_path = '/media/zzr/Data/skin_xml/img_result/'

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


def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[- 1].split('.ndpi')[0]
    return filename


def read_image(image_path, stride):
    try:
        image = OpenSlide(image_path)
        w, h = image.dimensions
        n = int(math.floor((h - 224) / stride + 1))
        m = int(math.floor((w - 224) / stride + 1))
        level = image.level_count - 1
        downsample_image = np.array(image.read_region((0, 0), level, image.level_dimensions[level]))
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

    return image, downsample_image, level, m, n


labels = ['背景', '表皮', '真皮', '脂肪', '毛囊', '汗腺']
img_WSI_all = []
img_WSI_all.append(os.path.join(img_WSI_dir, '2018-06-06 15.14.49.ndpi'))#train:2018-06-06 15.14.49.ndpi 2018-06-06 16.14.09.ndpi#test:2018-06-06 15.15.56.ndpi
for name1 in img_WSI_all:
    start = time.time()
    stride = 16 # set stride
    name = get_filename_from_path(name1)
    slide, downsample_image, level, m, n = read_image(name1, stride)
    mask = np.zeros([n, m])
    print '{} rows, {} columns'.format(n, m)
    print('%s Classification is in progress' % name1)
    patch_path = os.path.join(patch_save_path, name)
    ##
    nbatch = n // batchsize + 1
    for i in range(m):
        for k in range(nbatch):
            idx = np.arange(k * batchsize, min(n, (k + 1) * batchsize))
            temp = np.zeros([len(idx), 224, 224, 3])
            for tdx in idx:
                indexofdata = tdx % batchsize
                img_tmp = skimage.img_as_float(
                    np.array(slide.read_region((stride * i, stride * tdx), 0, (224, 224)))).astype(np.float32)[..., 0:-1]
                net.blobs['data'].data[indexofdata] = transformer.preprocess('data', img_tmp)
                temp[tdx-idx[0]] = img_tmp

            net.forward()
            for tdx in idx:
                indexofdata = tdx % batchsize
                prob = net.blobs['prob'].data[indexofdata].flatten()
                order = prob.argmax()
                print nbatch*i + k, '/', m*nbatch,  tdx, '/', idx[0]+len(idx)-1, 'the class is', labels[order], np.max(prob)
                # Path = patch_path + '/' + str(order)
                # if order == 1 or order == 4 or order == 5:
                #     if not os.path.exists(Path):
                #         os.makedirs(Path)
                #     io.imsave(os.path.join(Path, name[0:-4] + '_' + str(i) + '_' + str(tdx) + '.png'), temp[tdx-idx[0]])
                mask[tdx, i] = order

    end = time.time()
    np.save(save_file + 'Result_' + name + '.npy', mask)
    print('has done...')
    print(end - start)
All_end = time.time()
print(All_end - All_start)
print('success')