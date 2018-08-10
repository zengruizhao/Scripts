# coding=utf-8
import numpy as np
import os
import h5py
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import transform
from skimage.filters import threshold_otsu
from skimage.morphology import reconstruction, remove_small_objects, binary_erosion, disk, erosion, square
from skimage.measure import regionprops, find_contours, label
from skimage.util import crop
from PIL import Image
import time


def copy_file():#copy files from one path to another path
    path = '/home/zzr/git/3D_net/3Dunet_abdomen_cascade/results'
    out_pred_path = '/home/zzr/Data/pancreas/caffe_data/96_96_32/train_test/compare/test_mask_pred'
    out_orig_path = '/home/zzr/Data/pancreas/caffe_data/96_96_32/train_test/compare/test_mask_orig'
    file_list = os.listdir(path)
    for file in file_list:
        a = os.path.join(path, file)
        sub_path = [i for i in os.listdir(a) if 'stage1' in i]
        aa = os.path.join(a, sub_path[0])
        ##
        sub_sub_path_pred = [i for i in os.listdir(aa) if not os.path.isfile(os.path.join(aa, i))]
        sub_sub_path_orig = [i for i in os.listdir(aa) if 'label' in i]
        pred = os.path.join(aa, sub_sub_path_pred[0])
        img = os.listdir(pred)[0]
        shutil.copyfile(os.path.join(pred, img), os.path.join(out_pred_path, img))
        # shutil.copyfile(os.path.join(aa, sub_sub_path_orig[0]), os.path.join(out_orig_path, sub_sub_path_orig[0]))


def read_h5():#read h5 file and show
    file = h5py.File('/home/zzr/git/3D_net/3Dunet_abdomen_cascade/mine/deform/h5/iter_00000.h5', 'r')
    data = file['data']
    label = file['label']
    data = np.squeeze(data)
    label = np.squeeze(label)
    label = np.transpose(label, (2, 1, 0))
    data = np.transpose(data, (2, 1, 0))
    print label.shape
    print data.shape
    for i in range(label.shape[2]):
        plt.subplot(121)
        plt.imshow(label[..., i])
        plt.subplot(122)
        plt.imshow(data[..., i], cmap='gray')
        plt.show()


def statics_slice():    # statics how many slices
    path = '/media/zzr/Data/Task07_Pancreas/preprocess/crop_img'
    shape = []
    shape1 = []
    for file in os.listdir(path):
        img = nib.load(os.path.join(path, file))
        img = img.get_data()
        shape.append(img.shape[0])
        shape1.append(img.shape[1])
    print np.min(shape), np.max(shape), np.min(shape1), np.max(shape1)
    plt.hist(shape, 100)
    plt.figure()
    plt.hist(shape1, 100)
    plt.show()


def generate_hdf5_list(path):
    with open('./mine/train.txt', 'w') as file:
        for i in os.listdir(path):
            sub = os.listdir(os.path.join(path, i))[0]
            image = [image for image in os.listdir(os.path.join(path, i + '/' + sub)) if '.h5' in image]
            all_path = os.path.join(os.path.join(path, i + '/' + sub + '/' + image[0]))
            file.write(all_path + '\n')


def OtsuAndMorphology(input_path, output_path):
    file_list = os.listdir(os.path.join(input_path, 'imagesTr'))
    file_list = [i for i in file_list if 'nii' in i]
    for idx, i in enumerate(file_list):
        print idx
        img_orig, img_affine = read_nii(os.path.join(input_path+'/'+'imagesTr', i))
        mask_orig, mask_affine = read_nii(os.path.join(input_path+'/'+'labelsTr', i))
        img = (img_orig-np.min(img_orig))/(np.max(img_orig) - np.min(img_orig))
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
        bw_list = np.squeeze(crop_bw(label_))
        ##
        img_out = img[bw_list[0]:bw_list[3], bw_list[1]:bw_list[4], bw_list[2]:bw_list[5]]
        label_out = mask_orig[bw_list[0]:bw_list[3], bw_list[1]:bw_list[4], bw_list[2]:bw_list[5]]
        # save_nii(img_out, img_affine, os.path.join(output_path+'/'+'crop_img', i))
        # save_nii(label_out, mask_affine, os.path.join(output_path+'/'+'crop_label', i))
        for ii in range(img.shape[2]):
            # result = label_[..., ii]
            result = img_out[..., ii]
            plt.imshow(result, cmap='gray')
            plt.axis('off')
            # plt.imshow(transform.rotate(result, 90))
            plt.show()


def read_nii(path):
    img = nib.load(path)
    affine = img.affine
    img = img.get_data()
    return img, affine


def save_nii(img, affine, path):
    img = nib.Nifti1Image(img, affine)
    nib.save(img, path)


def crop_bw(data):
    x_min = []
    y_min = []
    z_min = []
    x_max = []
    y_max = []
    z_max = []
    array = np.array(np.where(data == 1))
    x_min.append(min(array[0, :]))
    y_min.append(min(array[1, :]))
    z_min.append(min(array[2, :]))
    x_max.append(max(array[0, :]))
    y_max.append(max(array[1, :]))
    z_max.append(max(array[2, :]))
    return [x_min, y_min, z_min, x_max, y_max, z_max]


def read_file(fpath):
    BLOCK_SIZE = 1024
    with open(fpath, 'rb') as f:
        while True:
            block = f.read(BLOCK_SIZE)
            if block:
                yield block
            else:
                return


def mnist_show():
    from keras.preprocessing.image import ImageDataGenerator
    from keras.datasets import mnist
    import matplotlib.pyplot as plt
    from keras import backend as K
    K.set_image_dim_ordering('th')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.fit(x_train)
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
        for i in xrange(0, 9):
            plt.subplot(330+1+i)
            plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.show()
        break


def roisize():
    x_roi, y_roi, z_roi = [], [], []
    path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/label'
    for idx, labelFile in enumerate(os.listdir(path)):
        img = nib.load(os.path.join(path, labelFile))
        label_data = img.get_data()
        # if len(np.unique(label_data)) != 2:
        #     print labelFile
        array = np.array(np.where(label_data == 1))
        x_roi.append(np.max(array[0, :]) - np.min(array[0, :]))
        y_roi.append(np.max(array[1, :]) - np.min(array[1, :]))
        z_roi.append(np.max(array[2, :]) - np.min(array[2, :]))

    print np.min(x_roi), np.min(y_roi), np.min(z_roi)
    plt.hist(x_roi, 40)
    plt.figure()
    plt.hist(y_roi, 40)
    plt.figure()
    plt.hist(z_roi, 20)
    plt.show()


def rotate():
    path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/image'
    img = os.path.join(path, 'pancreas_001.nii.gz')
    img, affine = read_nii(img)
    outImage = np.copy(img)
    for slice in np.arange(img.shape[2]):
        data = img[..., slice]
        out = Image.fromarray(data.astype('float32'))
        out = out.rotate(10)
        outImage[..., slice] = np.array(out)
    save_nii(outImage, affine, os.path.join('/home/zzr/Data/pancreas', 'image.nii.gz'))


if __name__ == '__main__':
    # copy_file()
    # read_h5()
    # statics_slice()
    # generate_hdf5_list('/home/zzr/git/3D_net/3Dunet_abdomen_cascade/data')
    # OtsuAndMorphology(input_path='/media/zzr/Data/Task07_Pancreas',
    #                   output_path='/media/zzr/Data/Task07_Pancreas/preprocess')
    # mnist_show()
    roisize()
    # rotate()

