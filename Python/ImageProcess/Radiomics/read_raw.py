# coding=utf-8
import SimpleITK as Sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from nilearn.image import resample_img, reorder_img
import nibabel as nib
from dipy.align.reslice import reslice
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, erosion
from skimage.measure import regionprops, label


Pancreas_path = '/run/user/1000/gvfs/smb-share:server=darwin-mi.local,' \
                    'share=data/Pancreas/Radiology/MICCAI2018/Training/Pancreas'
Tumor_path = '/run/user/1000/gvfs/smb-share:server=darwin-mi.local,' \
                 'share=data/Pancreas/Radiology/MICCAI2018/Training/Tumor'


def img_reslice(img_data, img_affine, img_spacing, new_spacing=(1., 1., 1.)):
    data, affine = reslice(img_data, img_affine, img_spacing, new_spacing, cval=np.min(img_data))
    return data, affine


def img_resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


def crop_bw(array):
    x_min = min(array[0, :])
    y_min = min(array[1, :])
    z_min = min(array[2, :])
    x_max = max(array[0, :])
    y_max = max(array[1, :])
    z_max = max(array[2, :])
    return [x_min, y_min, z_min, x_max, y_max, z_max]


def OtsuAndMorphology(img_orig):
    img = (img_orig - np.min(img_orig))/(np.max(img_orig) - np.min(img_orig))
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
    filled_area = []
    label_list = []
    for prop in props:
        filled_area.append(prop.area)
        label_list.append(prop.label)
    filled_area_sort = np.sort(filled_area)
    true_label = label_list[np.squeeze(np.argwhere(filled_area == filled_area_sort[-1]))]
    label_ = label_ == true_label
    array = np.array(np.where(label_ == 1))
    bw_list = np.squeeze(crop_bw(array))
    ##
    img_out = img[bw_list[0]:bw_list[3], bw_list[1]:bw_list[4], bw_list[2]:bw_list[5]]
    print img_out.shape
    for ii in range(img_out.shape[2]):
        # result = label_[..., ii]
        result = img_out[..., ii]
        plt.imshow(result, cmap='gray')
        plt.axis('off')
        # plt.imshow(transform.rotate(result, 90))
        plt.show()

    return img_out


def crop_voxel(img):
    temp = 10
    array = np.array(np.where(img > np.min(img)))
    bw_list = crop_bw(array)
    img_out = img[np.max([bw_list[0]-temp, 0]):np.min([bw_list[3]+temp, img.shape[0]]),
                  np.max([bw_list[1]-temp, 0]):np.min([bw_list[4]+temp, img.shape[1]]),
                  np.max([bw_list[2], 0]):np.min([bw_list[5], img.shape[2]])]
    # for ii in range(img_out.shape[2]):
    #     result = img_out[..., ii]
    #     plt.imshow(result, cmap='gray')
    #     plt.axis('off')
    #     # plt.imshow(transform.rotate(result, 90))
    #     plt.show()
    return img_out


def main():
    dimension = []
    for mhd in [mhd for mhd in os.listdir(Tumor_path) if mhd.endswith('mhd')]:
        print mhd
        mhd_file = os.path.join(Tumor_path, mhd)
        image = Sitk.ReadImage(mhd_file)
        width, height, depth, direction, origin, spacing = \
            image.GetWidth(), image.GetHeight(), image.GetDepth(), \
            image.GetDirection(), image.GetOrigin(), image.GetSpacing()
        # affine
        affine = np.diag(spacing)
        affine = np.row_stack((affine, np.zeros([1, 3])))
        affine = np.column_stack((affine, np.reshape([0, 0, 0, 1], [4, 1])))
        # data
        data = np.transpose(Sitk.GetArrayFromImage(image), (2, 1, 0))
        data2 = crop_voxel(data)
        out_data, affine2 = img_reslice(data2, affine, spacing, new_spacing=(1.0, 1.0, 1.0))
        print out_data.shape
        nib.save(nib.Nifti1Image(out_data, affine2), os.path.join('../tumor/', mhd[:-4] + 'nii'))

        # # dimension.append(image.shape[0])
    #     plt.imshow(np.squeeze(data[50, ...]), cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    #
    # # cv2.imwrite('1.png', np.squeeze(image[50, ...]))
    # plt.hist(dimension, bins=100)
    # plt.show()


def createMask():
    path = '/home/zzr/Data/Pancreas_OS/tumor'
    for i in os.listdir(path):
        print i
        nii_file = nib.load(os.path.join(path, i))
        affine = nii_file.affine
        data = nii_file.get_fdata()
        mask = np.zeros(data.shape)
        idx = np.array(np.where(data != np.min(data)))
        for j in xrange(idx.shape[-1]):
            mask[idx[0, j], idx[1, j], idx[2, j]] = 1
        nib.save(nib.Nifti1Image(mask, affine), os.path.join('../mask/', i))


if __name__ == '__main__':
    # main()
    createMask()