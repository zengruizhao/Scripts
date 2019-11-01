# coding=utf-8
from __future__ import division
import SimpleITK as sitk
from multiprocessing import Process
import pydicom
import numpy as np
import os
import h5py
# import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
import time
import cv2
import scipy.io as mat
import shutil
import nrrd
import scipy.io as sio
import zipfile


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
    path = '/media/zzr/Data/Task07_Pancreas/ChangHai/preprocess'
    shape0, shape1, shape2 = [], [], []
    for idx, file_ in enumerate(os.listdir(path)):
        print idx
        img = nib.load(os.path.join(path, file_))
        img_zoom = img.header.get_zooms()[:3]
        shape0.append(img.shape[0])
        shape1.append(img.shape[1])
        shape2.append(img.shape[2])

    print np.min(shape0), np.max(shape0), np.min(shape1), np.max(shape1), np.min(shape2), np.max(shape2)
    print np.mean(shape0), np.mean(shape1), np.mean(shape2)
    plt.hist(shape0, len(os.listdir(path))//2), plt.figure()
    plt.hist(shape1, len(os.listdir(path))//2), plt.figure()
    plt.hist(shape2, len(os.listdir(path))//2), plt.show()


def generate_hdf5_list(path):
    with open('./mine/train.txt', 'w') as file:
        for i in os.listdir(path):
            sub = os.listdir(os.path.join(path, i))[0]
            image = [image for image in os.listdir(os.path.join(path, i + '/' + sub)) if '.h5' in image]
            all_path = os.path.join(os.path.join(path, i + '/' + sub + '/' + image[0]))
            file.write(all_path + '\n')


def read_nii(path):
    img = nib.load(path)
    affine = img.affine
    img = img.get_fdata()
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


def elastic_deformation():
    img_path = '/media/zzr/Data/Task07_Pancreas/TCIA/preprocess/img'
    mask_path = '/media/zzr/Data/Task07_Pancreas/TCIA/preprocess/label'
    img = os.path.join(img_path, 'PANCREAS_0001.nii.gz')
    mask = os.path.join(mask_path, 'PANCREAS_0001.nii.gz')
    img, affine_img = read_nii(img)
    mask, affine_mask = read_nii(mask)
    # img, mask = elastic_transform(img, mask, img.shape[1] * 0.03)
    img = gussian_noise(img)
    print np.unique(mask)
    save_nii(img, affine_img, os.path.join('/home/zzr/Data/pancreas', 'image.nii.gz'))
    save_nii(mask, affine_mask, os.path.join('/home/zzr/Data/pancreas', 'label.nii.gz'))


def gussian_noise(image):
    gussian = np.random.normal(0., 0.1, image.shape)
    return np.add(image, gussian)


def elastic_transform(image, mask, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    channel = image.shape[-1]
    image = np.concatenate((image, mask), axis=-1)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_NEAREST)

    return image[..., 0:channel], image[..., channel:]


def translation():
    img_path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/image'
    mask_path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/label'
    img = os.path.join(img_path, 'pancreas_001.nii.gz')
    mask = os.path.join(mask_path, 'pancreas_001.nii.gz')
    img, affine_img = read_nii(img)
    mask, affine_mask = read_nii(mask)
    shape = img.shape
    shape_size = shape[:2]
    # M = np.float32([[1, 0], [0, 1], [0, 0]])
    # M = cv2.getAffineTransform(M, np.zeros_like(M))
    M = np.array([[1, 0, -50], [0, 1, 50]], np.float32)
    image = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_NEAREST)
    for i in xrange(shape[-1]):
        plt.imshow(image[..., i])
        plt.show()


def copy_folder():
    ct_pnet = mat.loadmat('/media/zzr/My Passport/430/MRI_PNET.mat')
    path = '/media/zzr/My Passport/430/430'
    out_path = '/media/zzr/My Passport/430/MRI_PNET'
    for i in xrange(ct_pnet['MRI'][:, 0].size):
        print '*' * 5, i
        case_name = ct_pnet['MRI'][i, 0][0]
        # print len(os.listdir(os.path.join(path, case_name)))
        # delete existed folder
        # if os.path.exists(os.path.join(out_path, case_name)):
        #     shutil.rmtree(os.path.join(out_path, case_name))
        start = time.time()
        shutil.copytree(os.path.join(path, case_name), os.path.join(out_path, case_name))
        print time.time() - start


def pnet_ct():  # dicom2nii
    path = '/media/zzr/My Passport/430/CT/CT_PNET'
    out_path = '/media/zzr/My Passport/430/CT/Preprocess_CT2'
    lack = []
    for idx, case in enumerate(os.listdir(path)):
        if idx == 0:
            # print case
            print idx
            case_path = os.path.join(path, case)
            for idxx, phase in enumerate(os.listdir(case_path)):
                # print phase
                phase_path = os.path.join(case_path, phase)
                dicom_file = [i for i in os.listdir(phase_path) if i.endswith('dcm')]
                label_file = [i for i in os.listdir(phase_path) if i.endswith('nrrd')]
                if label_file:
                    if 'P' in label_file[0]:
                        DATA = np.zeros([512, 512, len(dicom_file)])
                        label_data, label_options = nrrd.read(os.path.join(phase_path, label_file[0]))
                        label_data[label_data == 6] = 1
                        affine = label_options['space directions']
                        spaceOrigin = np.append(np.array(label_options['space origin']), 1)
                        affine = np.row_stack((affine, np.zeros([1, 3])))
                        affine = np.array(np.column_stack((affine, np.reshape([0, 0, 0, 1], [4, 1]))), np.float32)
                        # affine = np.fabs(np.array(np.column_stack((affine, np.reshape(spaceOrigin, [4, 1]))), np.float32))
                        dicom_slice = []
                        for img in dicom_file:
                            dicom_slice.append(int(img.split('.')[0]))

                        dicom_slice = sorted(dicom_slice)

                        for img in dicom_file:
                            dicom = pydicom.read_file(os.path.join(phase_path, img))
                            try:
                                data = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
                                # plt.imshow(data)
                                # plt.show()
                            except AttributeError:
                                data = np.zeros_like(data)
                                print '*' * 10, 'AttributeError'

                            except ValueError:
                                data = np.zeros_like(data)
                                print '*' * 10, 'ValueError'

                            try:  # Prevent missing dicom file
                                # DATA[..., dicom.InstanceNumber - 1] = np.squeeze(np.transpose(data[..., None], (2, 1, 0)))
                                # DATA[..., dicom_slice.index(int(img.split('.')[0]))] = np.squeeze(np.transpose(data[..., None], (2, 1, 0)))
                                # plt.imshow(data)
                                # plt.show()
                                DATA[..., dicom_slice.index(int(img.split('.')[0]))] = np.transpose(data)
                            except IndexError:
                                pass

                        DATA = DATA[::-1, ::-1, ::-1]
                        if not os.path.exists(os.path.join(out_path, case)):
                            os.makedirs(os.path.join(out_path, case))

                        nib.save(nib.Nifti1Image(DATA, affine),
                                 os.path.join(os.path.join(out_path, case), label_file[0][:-5] + '_img.nii.gz'))
                        # nib.save(nib.Nifti1Image(label_data, affine),
                        #          os.path.join(os.path.join(out_path, case), label_file[0][:-5] + '_label.nii.gz'))

                else:
                    lack.append(case)

    print lack


def pnet_mri():  # dicom2nii
    path = '/media/zzr/My Passport/newMRI/Raw'
    out_path = '/media/zzr/My Passport/newMRI/preprocess'
    lack = []
    for idx, case in enumerate(os.listdir(path)):
        if idx == 1:
            print idx, case
            case_path = os.path.join(path, case)
            for idxx, phase in enumerate(os.listdir(case_path)):
                if idxx >= 0:
                    # print phase
                    phase_path = os.path.join(case_path, phase)
                    dicom_file = [i for i in os.listdir(phase_path) if i.endswith('dcm')]
                    label_file = [i for i in os.listdir(phase_path) if i.endswith('zip')]
                    if label_file:
                        for i in label_file:
                            z = zipfile.ZipFile(os.path.join(phase_path, i), 'r')
                            z.extractall(phase_path)
                            i = i[:-4]
                            label_data, label_options = nrrd.read(os.path.join(phase_path, i))
                            label_data[label_data == 6] = 1
                            affine = generateAffine(label_options)
                            dicom_slice = []
                            for img in dicom_file:
                                temp = img.split('.d')[0]
                                if '.' in temp:
                                    indexx = int(temp.split('.')[-1])
                                else:
                                    indexx = int(temp)
                                dicom_slice.append(indexx)

                            sorted(dicom_slice)
                            dicom = pydicom.read_file(os.path.join(phase_path, img))
                            DATA = np.zeros([dicom.Columns, dicom.Rows, len(dicom_file)])
                            for img in dicom_file:
                                data = np.zeros_like(np.squeeze(label_data[..., 0]))
                                dicom = pydicom.read_file(os.path.join(phase_path, img))
                                try:
                                    data = dicom.pixel_array
                                    # data = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept

                                except AttributeError:
                                    data = np.zeros_like(data)
                                    print '*' * 10, 'AttributeError'

                                except ValueError:
                                    data = np.zeros_like(data)
                                    print '*' * 10, 'ValueError'

                                try:  # Prevent missing dicom file
                                    DATA[..., int(dicom.InstanceNumber) - 1] = np.squeeze(np.transpose(data[..., None], (2, 1, 0)))
                                    #
                                    # temp = img.split('.d')[0]
                                    # if '.' in temp:
                                    #     indexx = int(temp.split('.')[-1])
                                    # else:
                                    #     indexx = int(temp)
                                    # DATA[..., dicom_slice.index(indexx)] = np.squeeze(np.transpose(data[..., None], (2, 1, 0)))

                                except IndexError:
                                    pass

                            if not os.path.exists(os.path.join(out_path, case)):
                                os.makedirs(os.path.join(out_path, case))

                            nib.save(nib.Nifti1Image(DATA, affine),
                                     os.path.join(os.path.join(out_path, case), i[:-5] + '_img.nii.gz'))
                            nib.save(nib.Nifti1Image(label_data, affine),
                                     os.path.join(os.path.join(out_path, case), i[:-5] + '_label.nii.gz'))

                    else:
                        lack.append(case)

    print lack


def generateAffine(options):
    """
    :param options:
    :return:
    """
    affine = np.zeros((3, 3))
    for x, i in enumerate(options['space directions']):
        for y, j in enumerate(i):
            if y == 2:
                affine[x, y] = float(j)
            else:
                affine[x, y] = -float(j)

    affine = np.row_stack((affine, np.zeros([1, 3])))
    origin = []
    for x, i in enumerate(options['space origin']):
        if x == 2:
            origin.append(float(i))
        else:
            origin.append(-float(i))
    origin = np.reshape(np.append(origin, 1), (4, 1))
    affine = np.column_stack((affine, origin))

    return affine


def temp():
    path = '/media/zzr/My Passport/430/MRI/Preprocess_MRI/03534455_WANG YONG SHENG_MR/'
    img_file = [i for i in os.listdir(path) if 'img' in i]
    label_file = [i for i in os.listdir(path) if 'label' in i]
    for idx, i in enumerate(img_file):
        img = nib.load(os.path.join(path, i))
        img_data = img.get_fdata()
        label = nib.load(os.path.join(path, label_file[idx]))
        label_data = label.get_fdata()
        for j in range(img_data.shape[-1]):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img_data[..., j])
            ax2.imshow(label_data[..., -1])
            plt.show()


def pnet_ct_time():  # dicom2nii
    path = '/media/zzr/My Passport/430/MRI/MRI_PNET'
    # with open('time.txt', 'w') as file:
    #     for idx, case in enumerate(os.listdir(path)):
    #         print idx
    #         case_path = os.path.join(path, case)
    #         phase_path = os.path.join(case_path, os.listdir(case_path)[0])
    #         dicom_file = [i for i in os.listdir(phase_path) if i.endswith('dcm')]
    #         dicom = pydicom.read_file(os.path.join(phase_path, dicom_file[0]))
    #         dicom_time = dicom.AcquisitionDate
    #         file.write(case + ' ' + dicom_time + '\n')
    data = []
    for idx, case in enumerate(os.listdir(path)):
        print(idx)
        case_path = os.path.join(path, case)
        phase_path = os.path.join(case_path, os.listdir(case_path)[0])
        dicom_file = [i for i in os.listdir(phase_path) if i.endswith('dcm')]
        dicom = pydicom.read_file(os.path.join(phase_path, dicom_file[0]))
        dicom_time = str(int(dicom.AcquisitionDate))
        data.append(case + '+' + dicom_time)

    mat.savemat('/media/zzr/My Passport/newMRI/MRI_time.mat', {'data': data})


def dicom_spacing():  # dicom2nii
    path = '/media/zzr/My Passport/430/MRI/MRI_PNET'
    data = []
    for idx, case in enumerate(os.listdir(path)):
        print idx
        case_path = os.path.join(path, case)
        for i in os.listdir(case_path):
            nrrd = [j for j in os.listdir(os.path.join(case_path, i)) if j.endswith('nrrd') & ('T2' in j)]
            if nrrd:
                dicom_file = [jj for jj in os.listdir(os.path.join(case_path, i)) if jj.endswith('dcm')]
                dicom = pydicom.read_file(os.path.join(os.path.join(case_path, i), dicom_file[0]))
                spacing = str(dicom.PixelSpacing[0])
                data.append(case + '+' + spacing)

    mat.savemat('/media/zzr/My Passport/430/MRI/spacing_T2.mat', {'data': data})


def dicomInformation():
    path = '/media/zzr/My Passport/newMRI/Raw'
    data = []
    for idx, case in enumerate(os.listdir(path)):
        print(idx)
        case_path = os.path.join(path, case)
        for i in os.listdir(case_path):
            nrrd = [j for j in os.listdir(os.path.join(case_path, i)) if j.endswith('nrrd') & ('T1' in j)]
            if nrrd:
                dicom_file = [jj for jj in os.listdir(os.path.join(case_path, i)) if jj.endswith('dcm')]
                dicom = pydicom.read_file(os.path.join(os.path.join(case_path, i), dicom_file[0]))
                spacing = str(dicom.PixelSpacing[0])
                rows = str(dicom.Rows)
                columns = str(dicom.Columns)
                sliceThickness = str(dicom.SliceThickness)
                try:
                    sliceSpacing = str(dicom.SpacingBetweenSlices)
                except AttributeError:
                    sliceSpacing = str(dicom.SliceThickness)

                manufacturer = str(dicom.Manufacturer)
                data.append(case + '+' + spacing + '+' + rows + '+' + columns + '+'
                            + manufacturer + '+' + sliceThickness + '+' + sliceSpacing)
                break

    mat.savemat('/media/zzr/My Passport/newMRI/T1Information.mat', {'data': data})


def show_img():
    path = '/media/zzr/Data/Task07_Pancreas/Game/preprocess/img'
    imageAll = [i for i in sorted(os.listdir(path)) if i.endswith('nii.gz')]
    for i in imageAll:
        print i
        img = nib.load(os.path.join(path, i))
        img = img.get_fdata()
        print np.max(img), np.min(img)
        print img.shape
        arr = img.flatten()
        plt.hist(arr)
        plt.show()
        # for j in xrange(img.shape[-1]):
        #     print j
        #     plt.imshow(img[:, :, j])
        #     # mngr = plt.get_current_fig_manager()
        #     # mngr.window.SetGeometry(0, 0, 500, 500)
        #     plt.show()


def dicomread():
    path = '/home/zzr/Data/pancreas/script/models/TCIA/Pancreas-CT/PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667'
    for i in sorted(os.listdir(path)):
        print i
        dicom = pydicom.read_file(os.path.join(path, i))
        print dicom.InstanceNumber
        plt.imshow(dicom.pixel_array)
        plt.show()


def mat2nii():
    path = '/media/zzr/My Passport/430/MRI/BiasFieldCorrection1/03534455_WANG YONG SHENG_MR'
    for i in os.listdir(path):
        img = sio.loadmat(os.path.join(path, i))['img']
        print img.shape
        for j in xrange(img.shape[-1]):
            plt.imshow(img[..., j])
            plt.show()


if __name__ == '__main__':
    # copy_file()
    # read_h5()
    # statics_slice()
    # generate_hdf5_list('/home/zzr/git/3D_net/3Dunet_abdomen_cascade/data')
    # OtsuAndMorphology(input_path='/media/zzr/Data/Task07_Pancreas/ChangHai',
    #                   output_path='/media/zzr/Data/Task07_Pancreas/ChangHai')
    # mnist_show()
    # roisize()
    # rotate()
    # elastic_deformation()
    # translation()
    # pnet_ct()
    # temp()
    # copy_folder()
    # pnet_mri()
    pnet_ct_time()
    # dicom_spacing()
    # dicomInformation()
    # show_img()
    # dicomread()
    # # mat2nii()
    # for root, dirs, files in os.walk('/media/zzr/My Passport/newMRI'):
    #     for img in files:
    #         if img.endswith('nrrd'):
    #             print(os.path.join(root, img))