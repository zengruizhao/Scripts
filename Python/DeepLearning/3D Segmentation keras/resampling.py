# coding=utf-8
import nibabel as nib
import os
from dipy.align.reslice import reslice
import scipy.io as sio
from UselessBackgroundRemoving import pathExist


matpath = '/media/zzr/My Passport/430/MRI/BiasFieldCorrection'
niipath = '/media/zzr/My Passport/430/MRI/Preprocess_MRI'
outpath = '/media/zzr/My Passport/430/MRI/Resample'
for idx, case_ in enumerate(os.listdir(matpath)):
    if idx == 32:
        print idx
        case_mat = os.path.join(matpath, case_)
        case_nii = os.path.join(niipath, case_)
        a = os.listdir(case_mat)
        for j in os.listdir(case_mat):
            mat = sio.loadmat(os.path.join(case_mat, j))
            img = mat['img']
            label = mat['label']
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
