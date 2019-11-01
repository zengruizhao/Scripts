# coding=utf-8
from openslide import OpenSlide
from skimage.measure import regionprops, label
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from creatXML_ASAP_boundary import creat_xml
import cv2
import os

small = np.load('/media/zzr/Data/skin_xml/phase2/mask/lowlevel.npy')
xmlout = '/media/zzr/Data/skin_xml/phase2/mask/'
img_WSI_dir = '/home/zzr/Desktop/'
color = np.array([[255, 0, 0],   # 表皮
                  [0, 255, 0],   # 真皮
                  [0, 0, 255],   # 脂肪
                  [125, 125, 125],  # 汗腺
                  [255, 255, 255],  # 背景
                  [255, 0, 255]  # 毛囊
                  ])


def small2big():
    stride = 1
    m, n = small.shape
    plt.imshow(small)
    plt.show()
    big = np.ones([m*stride, n*stride, 3])*4
    for i in range(m):
        for j in range(n):
            order = small[i, j]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 0] = color[int(order), 0]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 1] = color[int(order), 1]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 2] = color[int(order), 2]
    print 'done'
    # np.save('/media/zzr/Data/skin_xml/mask_result/big.npy', big)
    # out = np.resize(big, [m*1, n*1, 3])
    io.imsave('/media/zzr/Data/skin_xml/mask_result/cd3.png', big.astype(np.float)/255)


def specific_area(label=1):
    """
    Extract specific area according to label
    :param label:
    :return:
    """
    stride = 36  # set stride 36
    using_level = 1  # 0: max level; 1
    scale_ = 4
    # mrophology operation
    small_like = np.zeros_like(small)
    small_like[np.where(small == label)] = 1
    # result = small_like
    shape = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(small_like), cv2.MORPH_OPEN, kernel)
    image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    # result = cv2.morphologyEx(result, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    s_m, s_n = result.shape
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(small_like)
    # ax2.imshow(image_close)
    # plt.show()
    #
    rows = np.where(result)[0]
    columns = np.where(result)[1]
    wsi = OpenSlide(os.path.join(img_WSI_dir, '2018-06-06 15.14.49.ndpi'))
    big = np.zeros((s_m * stride / scale_, s_n * stride / scale_, 3))
    for idx, i in enumerate(rows):
        print idx, '/', len(rows)
        img = np.array(wsi.read_region((columns[idx] * stride * 2 ** using_level, i * stride * 2 ** using_level),
                                       using_level, (stride, stride)))[..., 0:-1]
        # plt.imshow(img)
        # plt.show()
        resize_img = cv2.resize(img, (stride/scale_, stride/scale_))
        big[i * stride/scale_:(i + 1) * stride/scale_,
            columns[idx] * stride/scale_:(columns[idx] + 1) * stride/scale_, :] = resize_img

    io.imsave('/media/zzr/Data/skin_xml/mask_result/' + str(label) + '.png', big.astype(np.float) / 255)


def small2big_new():  #
    stride = 1
    m, n = small.shape
    big = np.ones([m*stride, n*stride, 3])*4
    # 清除掉表皮周围的脂肪
    small_like = np.zeros_like(small)
    small_like[np.where(small == 0)] = 1    # 表皮
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    image_dilate = cv2.morphologyEx(np.array(small_like), cv2.MORPH_DILATE, kernel)
    small_zhifang = np.zeros_like(small)
    small_zhifang[np.where(small == 2)] = 1     # 脂肪
    small_and = np.logical_and(small_zhifang, image_dilate)
    small[small_and == 1] = 0
    # clear small objects
    small2 = np.ones_like(small) * 4
    for i in xrange(6):
        print i
        if i != 4:
            small_like = np.zeros_like(small)
            small_like[np.where(small == i)] = 1
            #
            label_, label_num = label(np.uint8(small_like), return_num=True)
            props = regionprops(label_)
            filled_area = []
            label_list = []
            for prop in props:
                filled_area.append(prop.area)
                label_list.append(prop.label)

            filled_area_sort = np.sort(filled_area)
            if len(filled_area_sort) > 1:
                true_label = list(np.squeeze(np.argwhere(filled_area == filled_area_sort[0])) + 1)
                for j in true_label:
                    label_[label_ == j] = 0

                small2[label_ != 0] = i

            else:
                small2[small_like == 1] = i
            # morphology close to clear small objects
            small_like = np.zeros_like(small2)
            small_like[np.where(small2 == i)] = 1
            shape = (3, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
            # image_open = cv2.morphologyEx(np.array(small_like), cv2.MORPH_OPEN, kernel)
            image_close = cv2.morphologyEx(np.array(small_like), cv2.MORPH_CLOSE, kernel)
            # image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)
            # result = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
            creat_xml(image_close,
                      resultFile=os.path.join(xmlout, str(i)+'.xml'))
            small2[image_close == 1] = i

    for i in range(m):
        for j in range(n):
            order = small2[i, j]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 0] = color[int(order), 0]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 1] = color[int(order), 1]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 2] = color[int(order), 2]
    print 'done'
    io.imsave('/media/zzr/Data/skin_xml/phase2/mask/mask.png', big.astype(np.float)/255)


if __name__ == '__main__':
    # small2big()
    # specific_area(label=0)
    small2big_new()
