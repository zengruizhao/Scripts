# coding=utf-8
"""
author: Zengrui Zhao
data: 2019.1.8
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, erosion
from skimage.measure import regionprops, label
import time
import os
from multiprocessing import Process
import math
import nibabel as nib
import argparse
import warnings
warnings.filterwarnings('ignore')


class UselessBackgroundRemoving:
    def __init__(self):
        self.arg = self.arg_parse()
        self.inputPath = self.arg.inputPath
        self.outPath = self.arg.outPath
        self.nProcs = self.arg.numberOfProcess
        self.maskPath = self.arg.maskPath

    def arg_parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--inputPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA/Raw/img')
        parser.add_argument('--maskPath', default='/media/zzr/Data/Task07_Pancreas/TCIA/Raw/label')
        parser.add_argument('--outPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA')
        parser.add_argument('--numberOfProcess', type=int, default=1)
        arg_ = parser.parse_args()
        return arg_

    def thread_OtsuAndMorphology(self, start_name, end_name, all_file, mask_list=None):
        file_list = all_file[start_name:end_name]
        for idx, i in enumerate(file_list):
            print i
            img_orig, img_affine = self.read_nii(os.path.join(self.inputPath, i))
            mask_orig, mask_affine = None, None
            if self.maskPath:
                mask_orig, mask_affine = self.read_nii(os.path.join(self.maskPath, mask_list[idx]))
            # To make the affine of img equal to the affine of label
            # assert((img_affine == mask_affine).all())
            img = img_orig
            # img = (img_orig-np.min(img_orig))/(np.max(img_orig) - np.min(img_orig))
            # otsu threshold
            thresh = threshold_otsu(img)
            binary = img > thresh
            seed = np.copy(binary)
            seed[1:-1, 1:-1] = binary.max()
            mask = binary
            filled = reconstruction(seed, mask, method='erosion')
            # for i in xrange(filled.shape[2]):
            #     plt.imshow(img[..., i], cmap='gray')
            #     plt.axis('off')
            #     plt.show()
            # erosion
            erosion_ = erosion(filled)
            # bwlabel
            label_, label_num = label(np.uint8(erosion_), return_num=True)
            props = regionprops(label_)
            filled_area, label_list = [], []
            for prop in props:
                filled_area.append(prop.area)
                label_list.append(prop.label)
            filled_area_sort = np.sort(filled_area)
            true_label = label_list[int(np.squeeze(np.argwhere(filled_area == filled_area_sort[-1])))]
            label_ = label_ == true_label
            ##
            h_w_1_min = np.min(np.unique(np.where(label_)[0]))
            h_w_1_max = np.max(np.unique(np.where(label_)[0]))
            h_w_2_min = np.min(np.unique(np.where(label_)[1]))
            h_w_2_max = np.max(np.unique(np.where(label_)[1]))
            slice_min = np.min(np.unique(np.where(label_)[-1]))
            slice_max = np.max(np.unique(np.where(label_)[-1]))
            img_out = img[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
            nib.save(nib.Nifti1Image(img_out, img_affine),
                     os.path.join(pathExist(self.outPath + '/' + 'crop_img'), i))
            if self.maskPath:
                label_out = mask_orig[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
                nib.save(nib.Nifti1Image(label_out, mask_affine),
                         os.path.join(pathExist(self.outPath+'/'+'crop_label'), i))

            # for ii in range(img.shape[2]):
            #     # result = label_[..., ii]
            #     result = img_out[..., ii]
            #     plt.imshow(result, cmap='gray')
            #     plt.axis('off')
            #     # plt.imshow(transform.rotate(result, 90))
            #     plt.show()

    def OtsuAndMorphology(self):
        start = time.time()
        file_list = os.listdir(self.inputPath)
        # file_list = [i for i in file_list if 'nii' in i]
        mask_list = None
        if self.maskPath:
            mask_list = os.listdir(self.maskPath)
        ProcessPointer = [None] * self.nProcs
        img_name_per_core = int(math.ceil(len(file_list) / self.nProcs))
        for proc in xrange(self.nProcs):
            start_name = proc * img_name_per_core
            end_name = min((proc + 1) * img_name_per_core, len(file_list))
            ProcessPointer[proc] = Process(target=self.thread_OtsuAndMorphology, args=(
                start_name, end_name, file_list, mask_list))
            ProcessPointer[proc].start()

        for proc in xrange(self.nProcs):
            ProcessPointer[proc].join()

        print time.time() - start

    def read_nii(self, path):
        img = nib.load(path)
        affine = img.affine
        img = img.get_fdata()

        return img, affine


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path


if __name__ == '__main__':
    UselessBackgroundRemoving().OtsuAndMorphology()
