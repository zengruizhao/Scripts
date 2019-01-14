# coding=utf-8
"""
TCIA 82 Pancreas Segmentation
"""
import os
import nibabel as nib
import numpy as np
import pydicom
from UselessBackgroundRemoving import pathExist


path = '/media/zzr/My Passport/TCIA'
All_case = sorted(os.listdir(os.path.join(path, 'Pancreas-CT')))
mask_list = sorted(os.listdir(os.path.join(path, 'label'))[::-1])
for idx, case in enumerate(All_case):
    if idx >= 0:
        print case
        sub_path = os.path.join(os.path.join(os.path.join(path, 'Pancreas-CT'), case),
                                os.listdir(os.path.join(os.path.join(path, 'Pancreas-CT'), case))[0])
        sub_sub_path = os.path.join(sub_path, os.listdir(sub_path)[0])
        DATA = np.zeros([512, 512, len(os.listdir(sub_sub_path))])
        for img in os.listdir(sub_sub_path):
            dicom = pydicom.read_file(os.path.join(sub_sub_path, img))
            data = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
            DATA[..., dicom.InstanceNumber - 1] = np.transpose(data)

        label = nib.load(os.path.join(os.path.join(path, 'label'), mask_list[idx]))
        label_data = label.get_fdata()[::-1, :, ::-1]
        affine = np.fabs(label.affine)
        DATA = DATA[::-1, :, ::-1]
        nib.save(nib.Nifti1Image(DATA, affine),
                 os.path.join(pathExist(os.path.join(path, 'img')), case + '.nii.gz'))
        nib.save(nib.Nifti1Image(label_data, affine),
                 os.path.join(pathExist(os.path.join(path, 'label_new')), case + '.nii.gz'))