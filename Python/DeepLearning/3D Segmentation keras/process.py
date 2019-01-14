# -*- coding:utf-8 -*-
import nibabel as nib
import numpy as np
import os
from nilearn.image import resample_img, reorder_img
from dipy.align.reslice import reslice
import argparse
from UselessBackgroundRemoving import pathExist


def arg_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--imgPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA/crop/crop_img')
    parse.add_argument('--segPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA/crop/crop_label',
                       help='if one does not have label, set on None')
    parse.add_argument('--outImgPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA/preprocess1/img')
    parse.add_argument('--outLabelPath', type=str, default='/media/zzr/Data/Task07_Pancreas/TCIA/preprocess1/label')
    parse.add_argument('--targetShape', type=tuple, default=(256, 160, 48))
    parse.add_argument('--wMinimum', type=float, default=-100.)
    parse.add_argument('--wMaximum', type=float, default=200.)
    parse.add_argument('--interpolation', type=str, default='nearest')

    return parse.parse_args()


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


# 使得间距为1*1*1
def img_reslice(img_data, img_affine, img_zoom, new_zooms=(1., 1., 1.)):
    data, affine = reslice(img_data, img_affine, img_zoom, new_zooms)
    return data, affine


# delete slice of no label
def delete_slice(img_path,seg_path):
    img_names = os.listdir(img_path)
    img_names = [i for i in img_names if '.nii' in i]
    for img_name in img_names:
        img = nib.load(os.path.join(img_path, img_name))
        img_zoom = img.header.get_zooms()[:3]
        img_affine = img.affine
        img_data = img.get_data()
        seg = nib.load(os.path.join(seg_path, img_name))
        seg_data = seg.get_data()
        # img_data, img_affine = img_reslice(img_data, img_affine, img_zoom, new_zooms=(1., 1., 1.), order=1)
        # seg_data, seg_affine = img_reslice(seg_data, img_affine, img_zoom, new_zooms=(1., 1., 1.), order=0)
        array = np.array(np.where(seg_data == 1))
        z_min = min(array[2, :])
        z_max = max(array[2, :])
        img_data = img_data[..., z_min:z_max+1]
        seg_data = seg_data[..., z_min:z_max+1]
        img_data = nib.Nifti1Image(img_data, img_affine)
        seg_data = nib.Nifti1Image(seg_data, img_affine)
        print img_data.shape
        nib.save(img_data,
                 os.path.join('/home/zzr/Data/pancreas/delete_slice/image', img_name))
        nib.save(seg_data,
                 os.path.join('/home/zzr/Data/pancreas/delete_slice/label', img_name))


def crop_image(img_path, seg_path, range_list):
    img_names = os.listdir(img_path)
    x_min, x_max, y_min, y_max = range_list
    img_names = [i for i in img_names if '.nii' in i]
    for idx, img_name in enumerate(img_names):
        img = nib.load(os.path.join(img_path, img_name))
        img_data = img.get_data()
        img_affine = img.affine
        out_img = img_data[x_min:x_max, y_min:y_max, :]
        print out_img.shape
        out_img = nib.Nifti1Image(out_img, img_affine)
        seg = nib.load(os.path.join(seg_path, img_name))
        seg_affine = seg.affine
        seg_data = seg.get_data()
        out_seg = seg_data[x_min:x_max, y_min:y_max, :]
        out_seg = nib.Nifti1Image(out_seg, seg_affine)
        nib.save(out_img,
                os.path.join('/home/zzr/Data/pancreas/shrink_size/image', img_name))
        nib.save(out_seg,
                os.path.join('/home/zzr/Data/pancreas/shrink_size/label', img_name))


# resize图像
def img_resize(image, new_shape, interpolation="nearest"):
    """
    Reshape the raw image
    :param image:
    :param new_shape:
    :param interpolation:
    :return:
    """
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    # ras_image = image
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine,
                        target_shape=output_shape, interpolation=interpolation)


# 处理数据
def process_data(arg):
    img_names = sorted(os.listdir(arg.imgPath))
    # img_names = [i for i in img_names if '.nii' in i]
    for idx, img_name in enumerate(img_names):
        if idx >= 0:
            # img处理
            print img_name
            img = nib.load(os.path.join(arg.imgPath, img_name))
            img_affine = img.affine
            # img_zoom = img.header.get_zooms()[:3]
            img_data = img.get_fdata()
            # img_data, img_affine = img_reslice(img_data, img_affine, img_zoom, new_zooms=(1., 1., 1.))
            img_data = np.clip(img_data, arg.wMinimum, arg.wMaximum)
            img_data = normalize_image(img_data)
            img_data = (img_data - np.mean(img_data)) / np.std(img_data)
            img_data = img_resize(nib.Nifti1Image(img_data, img_affine), arg.targetShape, arg.interpolation)
            # seg处理
            if arg.segPath:
                seg = nib.load(os.path.join(arg.segPath, img_name))
                # seg_zoom = img.header.get_zooms()[:3]
                seg_affine = seg.affine
                seg_data = seg.get_data()
                # seg_data, seg_affine = img_reslice(seg_data, seg_affine, seg_zoom, new_zooms=(1., 1., 1.))
                seg_data[seg_data > 0] = 1  # only pancreas
                seg_data = img_resize(nib.Nifti1Image(seg_data, seg_affine), arg.targetShape, arg.interpolation)
                nib.save(seg_data, os.path.join(pathExist(arg.outLabelPath), img_name))

            nib.save(img_data, os.path.join(pathExist(arg.outImgPath), img_name))


# 这个函数是用来扩充到目标尺度
def pad_process(data, target_pad_shape=(144, 144, 80)):
    img_shape = data.shape
    x = img_shape[0]
    y = img_shape[1]
    z = img_shape[2]
    if x <= target_pad_shape[0]:
        pad = target_pad_shape[0] - x
        data = np.pad(data, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant')
    elif target_pad_shape[0] < x:
        additional = x - target_pad_shape[0]
        data = data[additional // 2:-(additional - additional // 2), :, :]

    if y <= target_pad_shape[1]:
        pad = target_pad_shape[1] - y
        data = np.pad(data, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant')
    elif target_pad_shape[1] < y:
        additional = y - target_pad_shape[1]
        data = data[:, additional // 2:-(additional - additional // 2), :]

    if z <= target_pad_shape[2]:
        pad = target_pad_shape[2] - z
        data = np.pad(data, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)), 'constant')
    elif target_pad_shape[2] < z:
        additional = z - target_pad_shape[2]
        data = data[:, :, additional // 2:-(additional - additional // 2)]
    return data


def main():
    arg = arg_parse()
    # delete_slice(img_path, seg_path)
    # crop_image(img_path, seg_path, [70, 370, 80, 380])
    process_data(arg)


if __name__ == '__main__':
    main()

