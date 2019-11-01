# coding=utf-8
import nibabel as nib
import os
import scipy.io as sio


matpath = '/media/zzr/My Passport/430/MRI/IntensityStandardization'
niipath = '/media/zzr/My Passport/430/MRI/Resample'
outpath = '/media/zzr/My Passport/430/MRI/IntensityStandardization_nii'


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path


for idx, case_ in enumerate(os.listdir(matpath)):
    if idx >= 0:
        print idx
        case_mat = os.path.join(matpath, case_)
        case_nii = os.path.join(niipath, case_)
        a = [i for i in os.listdir(case_mat) if i.endswith('mat')]
        for j in a:
            mat = sio.loadmat(os.path.join(case_mat, j))
            img = mat['img']
            label = mat['label']
            nii_name = [i for i in os.listdir(case_nii) if j.split('.')[0] in i and 'img' in i]
            nii = nib.load(os.path.join(case_nii, nii_name[0]))
            affine = nii.affine
            nib.save(nib.Nifti1Image(img, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                nii_name[0].split('_img')[0] + '_img.nii.gz'))
            nib.save(nib.Nifti1Image(label, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                  nii_name[0].split('_img')[0] + '_label.nii.gz'))

