# coding=utf-8
import nibabel as nib
import os
from dipy.align.reslice import reslice
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path


matpath = '/media/zzr/My Passport/newMRI/BiasFieldCorrection'
niipath = '/media/zzr/My Passport/newMRI/preprocess'
outpath = '/media/zzr/My Passport/newMRI/Resample'
for idx, case_ in enumerate(sorted(os.listdir(matpath))):
    if idx == 29:
        case_mat = os.path.join(matpath, case_)
        case_nii = os.path.join(niipath, case_)
        a = os.listdir(case_mat)
        for idxj, j in enumerate(os.listdir(case_mat)):
            if idxj == 0:
                mat = sio.loadmat(os.path.join(case_mat, j))
                img = mat['img']
                label = mat['label']
                label = label[..., :-1]
                # img = img[..., 11:label.shape[-1] + 11]
                # for jj in range(label.shape[-1]):
                #     print(jj)
                #     plt.subplot(121)
                #     plt.imshow(img[..., 11 + jj], cmap='gray')
                #     plt.subplot(122)
                #     plt.imshow(label[..., jj], cmap='gray')
                #     plt.show()
                nii_name = [i for i in os.listdir(case_nii) if j.split('.')[0] in i and 'img' in i]
                nii = nib.load(os.path.join(case_nii, nii_name[0]))
                affine = nii.affine
                zoom = nii.header.get_zooms()[:3]
                data, affine1 = reslice(img, affine, zoom, (0.66, 0.66, 0.66))
                label, affine = reslice(label, affine, zoom, (0.66, 0.66, 0.66))
                nib.save(nib.Nifti1Image(data, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                     nii_name[0].split('_img')[0] + '_img.nii.gz'))
                nib.save(nib.Nifti1Image(label, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                      nii_name[0].split('_img')[0] + '_label.nii.gz'))
