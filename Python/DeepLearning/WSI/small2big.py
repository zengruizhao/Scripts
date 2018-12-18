# coding=utf-8
from openslide import OpenSlide
from skimage.measure import regionprops, find_contours, label
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from creatXML_ASAP_boundary import creat_xml
from skimage import measure
from skimage.morphology import remove_small_objects, remove_small_holes
from xml.dom import minidom
from get_small_patch import hole_fill

xmlout = '/media/zzr/Data/skin_xml/mask_result/tif'
small = np.load('/media/zzr/Data/skin_xml/mask_result/tif.npy')
img_WSI_dir = '/media/zzr/Data/skin_xml/original_new/RawImage'
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
    io.imsave('/media/zzr/Data/skin_xml/mask_result/2018-06-06 151449_1.png', big.astype(np.float)/255)


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


def semantic2patch(label=0):
    stride = 36  # set stride 36
    using_level = 0  # 0: max level; 1
    patch = (256, 256)
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
    result = cv2.morphologyEx(result, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours = measure.find_contours(result, 0.5)
    casepath = os.path.join(img_WSI_dir, 'test.svs')
    wsi = OpenSlide(casepath)
    for i in xrange(len(contours)):
        print i, '/', len(contours)
        for j in range(len(contours[i])):
            y = int(contours[i][j, 0] * stride * 2 ** using_level)
            x = int(contours[i][j, 1] * stride * 2 ** using_level)
            img = np.array(wsi.read_region((x, y), using_level, patch))[..., 0:-1]
            savepath = os.path.join('/media/zzr/Data/skin_xml/semantic/', ''.join(os.path.basename(casepath).split(' ')))
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            io.imsave(os.path.join(savepath, str(x) + '_' + str(y) + '.png'), img.astype(np.float) / 255)


def semantic2whole(label=0):
    stride = 36  # set stride 36
    using_level = 0  # 0: max level; 1
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
    result = cv2.morphologyEx(result, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    s_m, s_n = result.shape
    result = remove_small_objects(result.astype(np.bool), 200)
    result = hole_fill(result)
    rows = np.where(result)[0]
    columns = np.where(result)[1]
    wsi = OpenSlide(os.path.join(img_WSI_dir, 'test.svs'))
    big = np.zeros((s_m * stride / scale_ + 256 / scale_, s_n * stride / scale_ + 256 / scale_, 3))
    # get the original area
    for idx, i in enumerate(rows):
        print idx, '/', len(rows)
        img = np.array(wsi.read_region((columns[idx] * stride * 2 ** using_level, i * stride * 2 ** using_level),
                                       using_level, (stride, stride)))[..., 0:-1]
        resize_img = cv2.resize(img, (stride / scale_, stride / scale_))
        big[i * stride / scale_:(i + 1) * stride / scale_,
        columns[idx] * stride / scale_:(columns[idx] + 1) * stride / scale_, :] = resize_img

    # get rid of other area
    segpath = '/media/zzr/Data/skin_xml/semantic_result/test.svs'
    contours = measure.find_contours(result, 0.5)
    for i in xrange(len(contours)):
        for j in range(len(contours[i])):
            y = int(contours[i][j, 0] * stride * 2 ** using_level)
            x = int(contours[i][j, 1] * stride * 2 ** using_level)
            img = io.imread(os.path.join(segpath, str(x) + '_' + str(y) + '_seg.png'))
            resize_img = cv2.resize(img, (256 / scale_, 256 / scale_))
            y_ = y / (scale_ * 2 ** using_level)
            x_ = x / (scale_ * 2 ** using_level)
            big[y_: y_ + 256 / scale_, x_: x_ + 256 / scale_, :][resize_img == 0] = 0

    io.imsave('/media/zzr/Data/skin_xml/mask_result/tif/tif_3.png', big.astype(np.float) / 255)


def semantic2xml(label=0, resultFile=''):
    stride = 36  # set stride 36
    using_level = 0  # 0: max level; 1
    scale_ = 4
    # mrophology operation
    small_like = np.zeros_like(small)
    small_like[np.where(small == label)] = 1
    shape = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape)
    image_open = cv2.morphologyEx(np.array(small_like), cv2.MORPH_OPEN, kernel)
    image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    s_m, s_n = result.shape
    result = remove_small_objects(result.astype(np.bool), 200)
    result = hole_fill(result)
    #
    rows = np.where(result)[0]
    columns = np.where(result)[1]
    wsi = OpenSlide(os.path.join(img_WSI_dir, 'test.svs'))
    big = np.zeros((s_m * stride / scale_ + 256 / scale_, s_n * stride / scale_ + 256 / scale_))
    # get the original area
    for idx, i in enumerate(rows):
        print idx, '/', len(rows)
        img = np.array(wsi.read_region((columns[idx] * stride * 2 ** using_level, i * stride * 2 ** using_level),
                                       using_level, (stride, stride)))[..., 0:-1]
        resize_img = cv2.resize(img, (stride / scale_, stride / scale_))
        big[i * stride / scale_:(i + 1) * stride / scale_,
        columns[idx] * stride / scale_:(columns[idx] + 1) * stride / scale_] = np.ones(resize_img.shape[0:-1])

    # get rid of other area
    segpath = '/media/zzr/Data/skin_xml/semantic_result/test.svs'
    contours = measure.find_contours(result, 0.5)
    for i in xrange(len(contours)):
        for j in range(len(contours[i])):
            y = int(contours[i][j, 0] * stride * 2 ** using_level)
            x = int(contours[i][j, 1] * stride * 2 ** using_level)
            img = io.imread(os.path.join(segpath, str(x) + '_' + str(y) + '_seg.png'))
            resize_img = cv2.resize(img, (256 / scale_, 256 / scale_))
            y_ = y / (scale_ * 2 ** using_level)
            x_ = x / (scale_ * 2 ** using_level)
            big[y_: y_ + 256 / scale_, x_: x_ + 256 / scale_][resize_img == 0] = 0

    # get xml
    big = remove_small_objects(big.astype(np.bool), 1000)
    big = hole_fill(big)
    # io.imsave('/media/zzr/Data/skin_xml/mask_result/tif/tif_3.png', big.astype(np.float))   # binary map
    contours = measure.find_contours(big, 0.5)
    np.save('contours.npy', contours)
    doc = minidom.Document()
    ASAP_Annotation = doc.createElement("ASAP_Annotations")
    doc.appendChild(ASAP_Annotation)
    Annotations = doc.createElement("Annotations")
    ASAP_Annotation.appendChild(Annotations)
    for i in xrange(len(contours)):
        Annotation = doc.createElement("Annotation")
        Annotation.setAttribute("Name", "_" + str(i))
        Annotation.setAttribute("Type", "Polygon")
        Annotation.setAttribute("PartOfGroup", "None")
        Annotation.setAttribute("Color", "#F4FA58")
        Annotations.appendChild(Annotation)
        Coordinates = doc.createElement("Coordinates")
        Annotation.appendChild(Coordinates)
        for j in range(len(contours[i])):
            y = int(contours[i][j, 0] * scale_) * 2 ** using_level
            x = int(contours[i][j, 1] * scale_) * 2 ** using_level
            Coordinate1 = doc.createElement("Coordinate")
            Coordinate1.setAttribute("Order", str(j))
            Coordinate1.setAttribute("Y", str(y))
            Coordinate1.setAttribute("X", str(x))
            Coordinates.appendChild(Coordinate1)

    f = file(resultFile, "w")
    doc.writexml(f)
    f.close()


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
                true_label = np.squeeze(np.argwhere(filled_area == filled_area_sort[0])) + 1
                for idx, j in enumerate(true_label):
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
    io.imsave('/media/zzr/Data/skin_xml/mask_result/tif/tif1.png', big.astype(np.float)/255)


if __name__ == '__main__':
    # small2big()
    # specific_area(label=0)
    # small2big_new()
    # semantic2patch(label=0)
    # semantic2whole()
    semantic2xml(label=0,
                 resultFile='/media/zzr/Data/skin_xml/mask_result/tif/tif.xml')
