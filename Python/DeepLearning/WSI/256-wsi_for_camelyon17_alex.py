#! coding=utf-8
caffe_root = '/home/hjxu/caffe-master/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
import time

import scipy
import cv2
import skimage

from scipy import misc
caffe.set_mode_gpu()
caffe.set_device(0)
import argparse
import glob
import os


start=time.time()

def get_bbox(cont_img, rgb_image=None):
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rgb_contour = None
    if rgb_image is not None:
        rgb_contour = rgb_image.copy()
        line_color = (255, 0, 0)  # blue color code
        cv2.drawContours(rgb_contour, contours, -1, line_color, 2)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes, rgb_contour


def read_wsi_tumor(wsi_path):
     try:
        wsi_image = OpenSlide(wsi_path)
        m, n = wsi_image.dimensions
        m, n = int(m / 256), int(n / 256)
        level_used = wsi_image.level_count-1
        if(level_used>=8):
            level_used = 8
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
        if(level_used<8):
            rgb_image = np.array(wsi_image.read_region((0, 0), level_used,
                                                       wsi_image.level_dimensions[level_used]))
            rgb_image = cv2.resize(rgb_image, (m,n))
     except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None, None, None, None

     return wsi_image, rgb_image, level_used,m,n


def find_roi_bbox(rgb_image):
    # hsv -> 3 channel
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    # mask -> 1 channel
    mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0

    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    # plt.imshow(image_open)
    # plt.show()
    bounding_boxes, rgb_contour = get_bbox(image_open, rgb_image=rgb_image)
    return bounding_boxes, rgb_contour, image_open

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename


start=time.time()


def test(WSI_path,save_path, npy_save_path):

    # wsi_name = get_filename_from_path(WSI_path)
    wsi_image, rgb_image, level,m,n=read_wsi_tumor(WSI_path)
    # image_heat_save_path = save_path + wsi_name + '_heatmap.jpg'
    bounding_boxes, rgb_contour, image_open = find_roi_bbox(np.array(rgb_image))
    image_heat_save = np.zeros((n, m))


    print('%s Classification is in progress' % WSI_path)
    for bounding_box in bounding_boxes:
        b_x_start = int(bounding_box[0])
        b_y_start = int(bounding_box[1])
        b_x_end = int(bounding_box[0]) + int(bounding_box[2])
        b_y_end = int(bounding_box[1]) + int(bounding_box[3])
        #        X = np.random.random_integers(b_x_start, high=b_x_end, size=500 )
        #        Y = np.random.random_integers(b_y_start, high=b_y_end, size=int((b_y_end-b_y_start)//2+1 ))
        col_cords = np.arange(b_x_start, b_x_end)
        row_cords = np.arange(b_y_start, b_y_end)
        mag_factor = 256
        #        for x, y in zip(X, Y):
        #            if int(tumor_gt_mask[y, x]) != 0:
        for x in col_cords:
            for y in row_cords:
                if int(image_open[y, x]) != 0:
                    x_large = x * mag_factor
                    y_large = y * mag_factor
                    patch = wsi_image.read_region((x_large, y_large), 0, (256, 256))
                    img_tmp = skimage.img_as_float(np.array(patch))
                    img1 = np.tile(img_tmp, (1, 1, 3))
                    img2 = img1[:, :, :3]
                    net.blobs['data'].data[...] = transformer.preprocess('data', img2)
                    out = net.forward()
                    prob = out['prob'][0][0]#指定转移位置的概率值
                    # print y, x
                    image_heat_save[y, x] = prob
    print save_path,'in save...'
    scipy.misc.imsave(save_path, image_heat_save)
    np.save(npy_save_path,image_heat_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--num1', type=int, default=None)
    parser.add_argument('--num2', type=int, default=None)

    args = parser.parse_args()


    #=============================================
   # root = '/home/hjxu/camelyon16/Alexnet/profile/'
    deploy = '/home/hjxu_disk/Camelyon16/patchs/normal_select/lmdb_aug/deploy.prototxt'
    model = '/home/hjxu_disk/Camelyon16/patchs/normal_select/model_alex_5/alex_5_iter_55000.caffemodel'

    mean_proto_path='/home/hjxu_disk/Camelyon16/patchs/normal_select/lmdb_5/train_mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data_mean = open(mean_proto_path, 'rb' ).read()
    blob.ParseFromString(data_mean)
    array = np.array(caffe.io.blobproto_to_array(blob))
    mean_npy  = array[0]

    # mean = '/home/hjxu_disk/Camelyon16/model_final/first_alexnet/train_mean.npy'

    net = caffe.Net(deploy, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mean_npy.mean(1).mean(1))
  #  transformer.set_mean('data', np.load(mean).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    caffe.set_mode_gpu()
    caffe.set_device(0)
    #================================================


    TUMOR_WSI_PATH = '/home/hjxu_disk/Camelyon17_data/test/'
    HEAT_MAP_SAVE_PATH = '/home/hjxu_disk/camelyon17_Image/heatmap/test_result_camelyon/'

    #===============================================
    wsi_folders = os.listdir(TUMOR_WSI_PATH)
    wsi_folders.sort()
    wsi_folders_arg = wsi_folders[args.num1: args.num2]
    for wsi_fold in wsi_folders_arg:
      if os.path.exists(HEAT_MAP_SAVE_PATH + wsi_fold):
          print (wsi_fold, 'has create,in error!!!!!!!!!!!!')
      else:
        os.makedirs(HEAT_MAP_SAVE_PATH + wsi_fold)
        wsi_paths = glob.glob(os.path.join(TUMOR_WSI_PATH + wsi_fold, '*.tif'))
        wsi_paths.sort()
        WSI_path = list(wsi_paths)

        for WSI_NAME in WSI_path:
            wsi_name = get_filename_from_path(WSI_NAME)
            heat_map_save_path = HEAT_MAP_SAVE_PATH + wsi_fold + '/' + wsi_name + '_heatmap.jpg'
            npy_save_path = HEAT_MAP_SAVE_PATH + wsi_fold + '/' + wsi_name + '_heatmap.npy'
            if os.path.exists(heat_map_save_path):
                continue
            test(WSI_NAME, heat_map_save_path, npy_save_path)
end = time.time()
print ('run time%s'%(end-start))
print('has done...')

