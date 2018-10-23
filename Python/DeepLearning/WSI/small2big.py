# coding=utf-8
from openslide import OpenSlide
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


small = np.load('/media/zzr/Data/skin_xml/mask_result/Result.npy')
img_WSI_dir = '/media/zzr/Data/skin_xml'
color = np.array([[255, 255, 255],  # 背景
                  [255, 0, 0],   # 表皮
                  [0, 255, 0],   # 真皮
                  [0, 0, 255],   # 脂肪
                  [255, 0, 255],  # 毛囊
                  [125, 125, 125]])  # 汗腺1


def small2big(small):
    stride = 1
    m, n = small.shape
    big = np.zeros([m*stride, n*stride, 3])
    for i in range(m):
        for j in range(n):
            order = small[i, j]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 0] = color[int(order), 0]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 1] = color[int(order), 1]
            big[(stride * i):stride * (i + 1), (stride * j):stride * (j + 1), 2] = color[int(order), 2]
    print 'done'
    # np.save('/media/zzr/Data/skin_xml/mask_result/big.npy', big)
    # out = np.resize(big, [m*1, n*1, 3])
    io.imsave('/media/zzr/Data/skin_xml/mask_result/out.png', big.astype(np.float)/255)


def specific_area(label=1):
    """
    Extract specific area according to label
    :param label:
    :return:
    """
    stride = 56
    mag_factor = 8
    # mrophology operation
    small_like = np.zeros_like(small)
    small_like[np.where(small == label)] = 1
    shape = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(small_like), cv2.MORPH_OPEN, kernel)
    image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    s_m, s_n = result.shape
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(small_like)
    # ax2.imshow(image_close)
    # plt.show()
    #
    big = np.zeros((s_m*stride/mag_factor, s_n*stride/mag_factor, 3))
    rows = np.where(result)[0]
    columns = np.where(result)[1]
    wsi = OpenSlide(os.path.join(img_WSI_dir, '2018-06-06 15.14.49.ndpi'))
    for idx, i in enumerate(rows):
        print idx, '/', len(rows)
        img = np.array(wsi.read_region((columns[idx] * stride, i * stride), 3, (stride, stride)))[..., 0:-1]
        plt.imshow(img)
        plt.show()
        resize_img = cv2.resize(img, (stride/mag_factor, stride/mag_factor))
        big[i * stride/mag_factor:(i+1) * stride/mag_factor,
            columns[idx] * stride/mag_factor:(columns[idx] + 1) * stride/mag_factor, :] = resize_img

    io.imsave('/media/zzr/Data/skin_xml/mask_result/' + str(label) + '.png', big.astype(np.float) / 255)


if __name__ == '__main__':
    # small2big(small)
    specific_area(label=5)