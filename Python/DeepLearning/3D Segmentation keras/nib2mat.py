# coding=utf-8
import nibabel as nib
import numpy as np
import os
import scipy.io as sio
from skimage.morphology import reconstruction, binary_erosion


def pathExist(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def temp():
    path = '/media/zzr/My Passport/430/Preprocess_new/02087575_JIN YI ZHONG_CT/'
    img_file = [i for i in os.listdir(path) if 'img' in i]
    img_file = [i for i in img_file if 'C-' in i]
    label_file = [i for i in os.listdir(path) if 'label' in i]
    label_file = [i for i in label_file if 'C-' in i]
    for idx, i in enumerate(img_file):
        img = nib.load(os.path.join(path, i))
        img_data = img.get_fdata()
        label = nib.load(os.path.join(path, label_file[idx]))
        label_data = label.get_fdata()
        h_w_1_min = np.min(np.unique(np.where(label_data)[0]))
        h_w_1_max = np.max(np.unique(np.where(label_data)[0]))
        h_w_2_min = np.min(np.unique(np.where(label_data)[1]))
        h_w_2_max = np.max(np.unique(np.where(label_data)[1]))
        slice_min = np.min(np.unique(np.where(label_data)[-1]))
        slice_max = np.max(np.unique(np.where(label_data)[-1]))
        out_data = img_data[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
        out_label = label_data[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
        sio.savemat('C-.mat', {'img': out_data, 'label': out_label})


def nib2mat_all():
    path = '/media/zzr/My Passport/430/Preprocess_new_new'
    out_path = '/media/zzr/My Passport/430/CT_PNET_Mat'
    for idxx, case in enumerate(os.listdir(path)):
        if idxx >= 0:
            print idxx, case
            case_file = os.path.join(path, case)
            img_file = [i for i in os.listdir(case_file) if 'img' in i]
            label_file = [i for i in os.listdir(case_file) if 'label' in i]
            for idx, i in enumerate(img_file):
                # if idx == 2:
                    assert i[:-11] == label_file[idx][:-13], 'img does not equal to label'
                    print '=' * 10, idx, i
                    img = nib.load(os.path.join(case_file, i))
                    img_data = img.get_fdata()
                    label = nib.load(os.path.join(case_file, label_file[idx]))
                    label_data = label.get_fdata()
                    h_w_1_min = np.min(np.unique(np.where(label_data)[0]))
                    h_w_1_max = np.max(np.unique(np.where(label_data)[0]))
                    h_w_2_min = np.min(np.unique(np.where(label_data)[1]))
                    h_w_2_max = np.max(np.unique(np.where(label_data)[1]))
                    slice_min = np.min(np.unique(np.where(label_data)[-1]))
                    slice_max = np.max(np.unique(np.where(label_data)[-1]))
                    out_data = img_data[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
                    out_label = label_data[h_w_1_min:(h_w_1_max + 1), h_w_2_min:(h_w_2_max + 1), slice_min:(slice_max + 1)]
                    if not os.path.exists(os.path.join(out_path, case)):
                        os.makedirs(os.path.join(out_path, case))

                    sio.savemat(os.path.join(os.path.join(out_path, case), i[:-11] + '.mat'),
                                {'img': out_data, 'label': out_label})


def nib2mat_all_wholeimage():
    path = '/media/zzr/My Passport/newMRI/preprocess'
    out_path = '/media/zzr/My Passport/newMRI/MatRaw'
    for idxx, case in enumerate(os.listdir(path)):
        if idxx == 1:
            print(idxx, case)
            case_file = os.path.join(path, case)
            img_file = [i for i in os.listdir(case_file) if 'img' in i]
            img_file = [i for i in img_file if 'T1' in i or 'T2' in i]
            # img_file = [i for i in img_file if 'T1' in i]
            # label_file = [i for i in os.listdir(case_file) if 'label' in i]
            for idx, i in enumerate(img_file):
                # if idx == 2:
                # assert i[:-11] == label_file[idx][:-13], 'img does not equal to label'
                print('=' * 10, idx, i)
                img = nib.load(os.path.join(case_file, i))
                img_data = img.get_fdata()
                label = nib.load(os.path.join(case_file, i[:-11] + '_label.nii.gz'))
                label_data = label.get_fdata()
                slice_min = np.min(np.unique(np.where(label_data)[-1]))
                slice_max = np.max(np.unique(np.where(label_data)[-1]))
                out_data = img_data[..., slice_min:(slice_max + 1)]
                out_label = label_data[..., slice_min:(slice_max + 1)]
                out_data = img_data
                out_label = label_data
                if not os.path.exists(os.path.join(out_path, case)):
                    os.makedirs(os.path.join(out_path, case))

                sio.savemat(os.path.join(os.path.join(out_path, case), i[:-11] + '.mat'),
                            {'img': out_data, 'label': out_label})


def mat2nib():
    matpath = '/media/zzr/My Passport/newMRI/IntensityStandardization'
    niipath = '/media/zzr/My Passport/430/MRI/Resample'
    outpath = '/media/zzr/My Passport/newMRI/IntensityStandardizationNii'
    for idx, case_ in enumerate(sorted(os.listdir(matpath))):
        if idx >= 0:
            print(idx, case_)
            case_mat = os.path.join(matpath, case_)
            case_nii = os.path.join(niipath, case_)
            for idxj, j in enumerate(os.listdir(case_mat)):
                if idxj >= 0:
                    mat = sio.loadmat(os.path.join(case_mat, j))
                    data = mat['img']
                    label = mat['mask']
                    nii_name = [i for i in os.listdir(case_nii) if j.split('.')[0] in i and 'img' in i]
                    nii = nib.load(os.path.join(case_nii, nii_name[0]))
                    affine = nii.affine
                    nib.save(nib.Nifti1Image(data, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                         nii_name[0].split('_img')[0] + '_img.nii.gz'))
                    nib.save(nib.Nifti1Image(label, affine), os.path.join(pathExist(os.path.join(outpath, case_)),
                                                                          nii_name[0].split('_img')[0] + '_label.nii.gz'))


def erode_shown():
    path = '/media/zzr/My Passport/430/Preprocess_new_new/D00744266_HONG YING_CT/'
    outpath = '/media/zzr/My Passport/430/erode/D00744266_HONG YING_CT/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    label_file = [i for i in os.listdir(path) if 'label' in i]
    label_file = [i for i in label_file if 'C-' in i]
    erode_filter = np.ones((5, 5))
    for idx, i in enumerate(label_file):
        label = nib.load(os.path.join(path, label_file[idx]))
        affine = label.affine
        label_data = label.get_fdata()
        label_erode = np.zeros_like(label_data)
        for j in range(label_data.shape[-1]):
            label_erode[..., j] = binary_erosion(label_data[..., j], selem=erode_filter)

        label_delete = np.logical_xor(label_erode, label_data)
        label_erode[label_delete == 1] = 2
        nib.save(nib.Nifti1Image(label_erode, affine), os.path.join(outpath, 'erode_label.nii.gz'))


if __name__ == '__main__':
    # temp()
    # nib2mat_all()
    # nib2mat_all_wholeimage()
    # erode_shown()
    mat2nib()