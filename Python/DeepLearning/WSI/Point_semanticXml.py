# coding=utf-8
import numpy as np
import cv2
import os
from skimage.measure import regionprops, label
from get_small_patch import hole_fill
from xml.dom import minidom
from collections import OrderedDict
from skimage import measure
from skimage.morphology import remove_small_objects
from openslide import OpenSlide
from skimage import io

color = OrderedDict([("r", "#ff0000"), ("g", "#00ff00"), ("b", "#0000ff"), ('y', '#ffff00')])
Channel = OrderedDict([('0', 'r'), ('1', 'g'), ('2', 'y')])
img_WSI_dir = '/media/zzr/Data/skin_xml/original_new/RawImage'
thumbnail_mask = np.load('/media/zzr/Data/skin_xml/mask_result/tif.npy')


def process(mask):
    mask = label(hole_fill(remove_small_objects(mask.astype(np.bool), 100)))
    mask = regionprops(mask)
    mask_coord = [i.centroid for i in mask]
    return mask_coord


def semantic2xml(label, doc, Annotations):
    stride = 36  # set stride 36
    using_level = 0  # 0: max level; 1
    scale_ = 4
    # mrophology operation
    small_like = np.zeros_like(thumbnail_mask)
    small_like[np.where(thumbnail_mask == label)] = 1
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
    contours = measure.find_contours(big, 0.5)
    for i in xrange(len(contours)):
        Annotation = doc.createElement("Annotation")
        Annotation.setAttribute("Name", 'Semantic_' + str(i))
        Annotation.setAttribute("Type", "Polygon")
        Annotation.setAttribute("PartOfGroup", "None")
        Annotation.setAttribute("Color", "#aa5500")
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

    return doc


def main():
    path = '/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data/Ultrasound Nerve Segmentation/temp/result'
    doc = minidom.Document()
    ASAP_Annotation = doc.createElement("ASAP_Annotations")
    doc.appendChild(ASAP_Annotation)
    Annotations = doc.createElement("Annotations")
    ASAP_Annotation.appendChild(Annotations)

    for ii, name in enumerate([i for i in os.listdir(path) if i.endswith('.png')]):
        print ii, '/', len(os.listdir(path))
        row = int(name.split('.')[0].split('_')[1]) - 1
        column = int(name.split('.')[0].split('_')[-1]) - 1
        mask = cv2.imread(os.path.join(path, name))
        red, green, blue = mask[..., 0], mask[..., 1], mask[..., 2]
        red_coord = process(red)
        green_coord = process(green)
        blue_coord = process(blue)
        for channel, coord in enumerate([red_coord, green_coord, blue_coord]):
            color_channel = Channel[str(channel)]
            for idx, j in enumerate(coord):
                Annotation = doc.createElement("Annotation")
                Annotation.setAttribute("Name", "Dot_" + str(idx))
                Annotation.setAttribute("Type", "Dot")
                Annotation.setAttribute("PartOfGroup", "None")
                Annotation.setAttribute("Color", color[color_channel])
                Annotations.appendChild(Annotation)
                Coordinates = doc.createElement("Coordinates")
                Annotation.appendChild(Coordinates)
                ##############################
                y = j[0] + 1024 * row
                x = j[1] + 1024 * column
                Coordinate1 = doc.createElement("Coordinate")
                Coordinate1.setAttribute("Order", '0')
                Coordinate1.setAttribute("Y", str(y))
                Coordinate1.setAttribute("X", str(x))
                Coordinates.appendChild(Coordinate1)

    print 'Dot done'

    doc = semantic2xml(0, doc, Annotations)
    print 'Semantic done'
    f = file('/run/user/1000/gvfs/afp-volume:host=Darwin-MI.local,user=zrzhao,volume=Data/Ultrasound Nerve Segmentation/temp/semantic_dot.xml', "w")
    doc.writexml(f)
    f.close()
    print 'Done'


if __name__ == '__main__':
    main()