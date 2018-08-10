# -*- coding:utf-8 -*-
import nibabel as nib
import numpy as np
import os
from nilearn.image import resample_img, reorder_img
from dipy.align.reslice import reslice

##
win_min = -100  # -145
win_max = 200   # 335
# 归一化
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
def img_resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


# 处理数据
def process_data(img_path, seg_path):
    img_names = os.listdir(img_path)
    img_names = [i for i in img_names if '.nii' in i]
    for idx, img_name in enumerate(img_names):
        # img处理
        print img_name
        target_resize_shape = (256, 160, 48)
        img = nib.load(os.path.join(img_path, img_name))
        img_affine = img.affine
        img_zoom = img.header.get_zooms()[:3]
        img_data = img.get_data()
        img_data, img_affine = img_reslice(img_data, img_affine, img_zoom, new_zooms=(1., 1., 1.))
        img_data[img_data < win_min] = win_min
        img_data[img_data > win_max] = win_max
        img_data = normalize_image(img_data)
        img_data = (img_data - np.mean(img_data)) / np.std(img_data)
        img_data = img_resize(nib.Nifti1Image(img_data, img_affine), target_resize_shape,interpolation="nearest")
        # img_data = pad_process(img_data)
        # img_data = nib.Nifti1Image(img_data, img_affine)
        # seg处理
        seg = nib.load(os.path.join(seg_path, img_name))
        seg_zoom = img.header.get_zooms()[:3]
        seg_affine = seg.affine
        seg_data = seg.get_data()
        seg_data, seg_affine = img_reslice(seg_data, seg_affine, seg_zoom, new_zooms=(1., 1., 1.))
        seg_data[seg_data > 0] = 1  # only pancreas
        seg_data = img_resize(nib.Nifti1Image(seg_data, seg_affine), target_resize_shape, interpolation="nearest")
        # combine_img_seg = np.concatenate((img,seg),axis=2)#这个在crop的时候比较好用
        # seg_data = pad_process(seg_data)
        print idx, seg_data.shape
        # seg_data = nib.Nifti1Image(seg_data, img_affine)
        # seg_data = img_resize(nib.Nifti1Image(seg_data, seg_affine), target_resize_shape,interpolation="nearest")
        # nib.save(img_data,
        #          os.path.join('/home/zzr/Data/pancreas/caffe_data/256_160_48/image', img_name))
        # nib.save(seg_data,
        #          os.path.join('/home/zzr/Data/pancreas/caffe_data/256_160_48/label', img_name))
    # return img,seg,img_affine,seg_affine


# 这个函数是用来扩充到目标尺度
def pad_process(data,target_pad_shape=(144, 144, 80)):
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
    img_path = '/media/zzr/Data/Task07_Pancreas/preprocess/crop_img/'
    seg_path = '/media/zzr/Data/Task07_Pancreas/preprocess/crop_label/'
    # delete_slice(img_path, seg_path)
    # crop_image(img_path, seg_path, [70, 370, 80, 380])
    process_data(img_path, seg_path)


if __name__ == '__main__':
    main()

