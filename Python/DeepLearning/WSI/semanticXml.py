# coding=utf-8
"""
author: Zengrui Zhao
data: 2018.12.17
"""
import numpy as np
import cv2
from openslide import OpenSlide
import os
from skimage import measure, io
from skimage.morphology import remove_small_objects
from get_small_patch import hole_fill
from xml.dom import minidom


class SemanticXml:
    def __init__(self, thumbnail_mask, label, img_WSI_dir, name):
        self.stride = 36
        self.using_level = 1
        self.scale_ = 4  # 4
        self.img_WSI_dir = img_WSI_dir
        self.name = name
        self.patch = (256, 256)
        self.thumbnail_mask = thumbnail_mask
        self.label = label
        self.result = np.zeros_like(thumbnail_mask)
        self.thumbnail_preprocess()

    def thumbnail_preprocess(self):
        # mrophology operation
        thumbnail_mask_like = np.zeros_like(self.thumbnail_mask)
        thumbnail_mask_like[np.where(self.thumbnail_mask == self.label)] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image_open = cv2.morphologyEx(np.array(thumbnail_mask_like), cv2.MORPH_OPEN, kernel)
        image_close = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
        image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(image_open, cv2.MORPH_CLOSE, kernel)
        self.result = cv2.morphologyEx(result, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    def semantic2patch(self, patch_path):
        """
        Extract the patch of boundary
        :param patch_path: the path of save
        :return:
        """
        contours = measure.find_contours(self.result, 0.5)
        casepath = os.path.join(self.img_WSI_dir, self.name)
        wsi = OpenSlide(casepath)
        for i in xrange(len(contours)):
            print i, '/', len(contours)
            for j in range(len(contours[i])):
                y = int(contours[i][j, 0] * self.stride * 2 ** self.using_level)
                x = int(contours[i][j, 1] * self.stride * 2 ** self.using_level)
                img = np.array(wsi.read_region((x, y), self.using_level, self.patch))[..., 0:-1]
                savepath = os.path.join(patch_path, ''.join(os.path.basename(casepath).split(' ')))
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                io.imsave(os.path.join(savepath, str(x) + '_' + str(y) + '.png'), img.astype(np.float) / 255)

        print 'semantic2patch done'

    def semantic2whole(self, mask_path=None, seg_path=None):
        s_m, s_n = self.result.shape
        self.result = hole_fill(remove_small_objects(self.result.astype(np.bool), 200))
        rows = np.where(self.result)[0]
        columns = np.where(self.result)[1]
        wsi = OpenSlide(os.path.join(self.img_WSI_dir, self.name))
        big = np.zeros((s_m * self.stride / self.scale_ + 256 / self.scale_,
                        s_n * self.stride / self.scale_ + 256 / self.scale_, 3))
        # get the original area
        for idx, i in enumerate(rows):
            print idx, '/', len(rows)
            img = np.array(wsi.read_region(
                (columns[idx] * self.stride * 2 ** self.using_level, i * self.stride * 2 ** self.using_level),
                self.using_level, (self.stride, self.stride)))[..., 0:-1]
            resize_img = cv2.resize(img, (self.stride / self.scale_, self.stride / self.scale_))
            big[i * self.stride / self.scale_:(i + 1) * self.stride / self.scale_,
            columns[idx] * self.stride / self.scale_:(columns[idx] + 1) * self.stride / self.scale_, :] = resize_img

        # get rid of other area
        contours = measure.find_contours(self.result, 0.5)
        for i in xrange(len(contours)):
            for j in range(len(contours[i])):
                y = int(contours[i][j, 0] * self.stride * 2 ** self.using_level)
                x = int(contours[i][j, 1] * self.stride * 2 ** self.using_level)
                img = io.imread(os.path.join(seg_path, str(x) + '_' + str(y) + '_seg.png'))
                resize_img = cv2.resize(img, (256 / self.scale_, 256 / self.scale_))
                y_ = y / (self.scale_ * 2 ** self.using_level)
                x_ = x / (self.scale_ * 2 ** self.using_level)
                big[y_: y_ + 256 / self.scale_, x_: x_ + 256 / self.scale_, :][resize_img == 0] = 0

        io.imsave(os.path.join(mask_path, 'tif1.png'), big.astype(np.float) / 255)
        print 'semantic2whole done'

    def semantic2xml(self, resultFile=None, seg_path=None):
        """

        :param resultFile:
        :param seg_path: the result of segnet
        :return:
        """
        s_m, s_n = self.result.shape
        self.result = hole_fill(remove_small_objects(self.result.astype(np.bool), 200))
        rows = np.where(self.result)[0]
        columns = np.where(self.result)[1]
        big = np.zeros((s_m * self.stride / self.scale_ + 256 / self.scale_,
                        s_n * self.stride / self.scale_ + 256 / self.scale_))
        # get the original area
        for idx, i in enumerate(rows):
            # print idx, '/', len(rows)
            big[i * self.stride / self.scale_:(i + 1) * self.stride / self.scale_,
            columns[idx] * self.stride / self.scale_:(columns[idx] + 1) * self.stride / self.scale_] = np.ones([self.stride / self.scale_,
                                                                                                               self.stride / self.scale_])

        # get rid of other area
        contours = measure.find_contours(self.result, 0.5)
        for i in xrange(len(contours)):
            for j in range(len(contours[i])):
                y = int(contours[i][j, 0] * self.stride * 2 ** self.using_level)
                x = int(contours[i][j, 1] * self.stride * 2 ** self.using_level)
                img = io.imread(os.path.join(seg_path, str(x) + '_' + str(y) + '_seg.png'))
                resize_img = cv2.resize(img, (256 / self.scale_, 256 / self.scale_))
                y_ = y / (self.scale_ * 2 ** self.using_level)
                x_ = x / (self.scale_ * 2 ** self.using_level)
                big[y_: y_ + 256 / self.scale_, x_: x_ + 256 / self.scale_][resize_img == 0] = 0

        # get xml
        big = hole_fill(remove_small_objects(big.astype(np.bool), 1000))
        # io.imsave('/media/zzr/Data/skin_xml/mask_result/tif/tif_3.png', big.astype(np.float))   # binary map
        contours = measure.find_contours(big, 0.5)
        # np.save('contours.npy', contours)
        if resultFile:
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
                    y = int(contours[i][j, 0] * self.scale_) * 2 ** self.using_level
                    x = int(contours[i][j, 1] * self.scale_) * 2 ** self.using_level
                    Coordinate1 = doc.createElement("Coordinate")
                    Coordinate1.setAttribute("Order", str(j))
                    Coordinate1.setAttribute("Y", str(y))
                    Coordinate1.setAttribute("X", str(x))
                    Coordinates.appendChild(Coordinate1)

            f = file(resultFile, "w")
            doc.writexml(f)
            f.close()

        print 'semantic2xml done'
        return big


def main():
    thumbnail_mask = np.load('/media/zzr/Data/skin_xml/phase2/mask/lowlevel.npy')
    semanticxml = SemanticXml(thumbnail_mask, label=0,
                              img_WSI_dir='/home/zzr/Desktop', name='2019-02-28 15.44.02.ndpi')
    # semanticxml.semantic2patch(patch_path='/media/zzr/Data/skin_xml/phase2/semantic/')
    # semanticxml.semantic2whole(mask_path='/media/zzr/Data/skin_xml/mask_result/tif',
    #                            seg_path='/media/zzr/Data/skin_xml/semantic_result/test.svs')
    semanticxml.semantic2xml(resultFile='/media/zzr/Data/skin_xml/phase2/mask/result.xml',
                             seg_path='/media/zzr/Data/skin_xml/phase2/semantic/result')


if __name__ == '__main__':
    main()