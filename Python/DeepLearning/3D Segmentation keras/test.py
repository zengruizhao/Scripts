# coding=utf-8
import nibabel as nib
import os
import numpy as np
from dipy.align.reslice import reslice
from utils import split_filename
from nilearn.image import resample_img, reorder_img, new_img_like
from model import Nest_Net, dice_coef_loss, dice_coef
from config import config
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy import misc
#from denseinference import CRFProcessor
from medpy import metric
from scipy import ndimage


def zoom(seg_data, zoom_factor=(1, 1, 1)):
    # new_img = ndimage.zoom(img_data, zoom_factor, order=4)
    # print 'new_img:' + str(new_img.shape)
    new_seg = ndimage.zoom(seg_data, zoom_factor, order=0)
    new_seg = new_seg.astype(np.uint8)
    print 'new_seg:' + str(new_seg.shape)
    return new_seg


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


def img_resize(image, new_shape, interpolation="continuous"):
    input_shape = np.asarray(image.shape, dtype=np.float16)
    ras_image = reorder_img(image, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def img_reslice(img_data, img_affine, img_zoom, new_zooms=(1., 1., 1.), order=1):
    data, affine = reslice(img_data, img_affine, img_zoom, new_zooms)
    return data, affine


def imshow(*args, **kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title = kwargs.get('title', '')
    axis_enabled = kwargs.get('axis', True)

    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        if not axis_enabled:
            plt.axis('off')
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap] * n
        if type(title) == str:
            title = [title] * n
        plt.figure(figsize=(n * 5, 10))
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.title(title[i])
            if not axis_enabled:
                plt.axis('off')
            plt.imshow(args[i], cmap[i], interpolation='none')
    plt.show()


def test():
    # img_path = '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/img'
    # seg_path = '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/seg'
    # img_pre = 'volume-'
    # seg_pre = 'segmentation-'
    # # save_path = '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/result'
    # # save_path = '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/second_picture'
    # save_path = '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/myresult/pixelres/very_good_third'
    #
    # img_name = 'volume-119.npy'
    # seg_name = 'segmentation-119.npy'

    # x,y = load_test_data()
    x = nib.load('/media/mw/mw_research/data/ruxian/3D_fudan/data/img/leftBao_Qiu_Sao_10499218.nii.gz')
    x = x.get_data()
    y = nib.load('/media/mw/mw_research/data/ruxian/3D_fudan/data/mask/leftBao_Qiu_Sao_10499218.nii.gz')
    y = y.get_data()
    # x = np.load(os.path.join(img_path, img_name))
    # y = np.load(os.path.join(seg_path, seg_name))
    # x=np.load('/home/mi/smj/data/liver/reslice_hu_norm_zs_400_random_np_test/img/volume-230_121.npy')
    # y=np.load('/home/mi/smj/data/liver/reslice_hu_norm_zs_400_random_np_test/seg/segmentation-230_121.npy')

    x = x[np.newaxis, :, :, :, np.newaxis]
    y = y[np.newaxis, :, :, :, np.newaxis]

    # model = newmodel.multi_loss_unet((128, 128, 128, 1), 2)
    # model.load_weights('./model/multi_loss/check_point.hdf5')

    # model = unet_model_3d((128, 128, 128, 1), 2)

    # model.load_weights('./model/good_1/3d_unet_model.h5')
    # model.load_weights('./model/resize_new/3d_unet_model.h5')

    model = unet_model_3d((160,240,96,1), 2)
    model.load_weights('./3d_unet_model.h5')

    # model = newmodel.unet_model_3d_5((128, 128, 128, 1), 2)
    # model.load_weights('./model/kernel_5/check_point.hdf5')

    # model = newmodel.unet_model_3d_5((128, 128, 128, 1), 2)
    # model.load_weights('./model/kernel_5_augment/check_point.hdf5')

    pred = model.predict(x, batch_size=1, verbose=1)
    # pred = np.array(pred)[1]
    # print np.where(np.squeeze(pred) > 0.3)+
    # np.save('imgs_mask_test.npy', imgs_mask_test)
    # imshow(np.squeeze(x), np.squeeze(y), (np.squeeze(pred)>0.5).astype(np.uint8), title=['Slice','Ground truth', 'Prediction'])
    # result = np.argmax(np.squeeze(pred), axis=-1).astype(np.uint8)
    result = np.array(np.squeeze(pred) > 0.7).astype(np.uint8)
    # imshow(np.squeeze(x)[70],np.squeeze(y)[70],result[70],title=['Slice','Ground truth', 'Prediction'])

    orl_img = np.squeeze(x)
    orl_seg = np.squeeze(y)

    # orl_img = nib.Nifti1Image(orl_img, np.eye(4,4))
    # orl_seg = nib.Nifti1Image(orl_seg, np.eye(4, 4))
    # result = nib.Nifti1Image(result, np.eye(4, 4))

    # nib.save(orl_img, '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/myresult/third_good/3D/orl_img.nii')
    # nib.save(orl_seg, '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/myresult/third_good/3D/orl_seg.nii')
    # nib.save(result, '/home/mi/smj/data/liver/3DUnet_data/get128128128shape_test/myresult/third_good/3D/result.nii')

    orl_img = np.squeeze(x)
    orl_seg = np.squeeze(y) * 255
    result = result * 255
    assert orl_img.shape == orl_seg.shape
    assert result.shape == orl_img.shape
    save_path = '/media/mw/mw_research/data/code/python/sunmingjian_lung_segmentation/data/test/1/'
    img_name = 'Pat1_T2_N4_Or'

    for i in range(orl_img.shape[2]):
        orl_img_name = os.path.join(save_path, 'img/',
                                    'img' + '_' + str(i) + img_name + '.png')
        orl_seg_name = os.path.join(save_path, 'seg/',
                                    'seg' + '_' + str(i) + img_name + '.png')
        pred_name = os.path.join(save_path, 'pred/',
                                 'pred' + '_' + str(i) + img_name + '.png')
        misc.imsave(orl_img_name, orl_img[:,:,i].transpose(1,0))
        print orl_img_name
        misc.imsave(orl_seg_name, orl_seg[:,:,i].transpose(1,0))
        print orl_seg_name
        misc.imsave(pred_name, result[:,:,i].transpose(1,0))
        print pred_name


def test_train(img_path, save_path):
    img_names = os.listdir(img_path)
    model = Nest_Net(shape=(None, None, None, 1), classes=2)
    model.load_weights('/home/zzr/Data/pancreas/script/models/pancreas_weights.060.h5')
    for img_name in img_names:
        print img_name
        img = nib.load(os.path.join(img_path, img_name))
        img_data = img.get_data()
        affine = img.affine
        x = img_data[np.newaxis, :, :, :, np.newaxis]
        pred = model.predict(x, batch_size=1, verbose=1)
        result = np.array(np.squeeze(pred[..., -1] > 0.5)).astype(np.uint8)
        print result.shape
        # img_final = ndimage.rotate(img_final, 180, (2, 1))
        img_final = nib.Nifti1Image(result, affine)
        nib.save(img_final, os.path.join(save_path, img_name))


def evaluate_test(img_path, mask_path):
    img_names = os.listdir(img_path)
    model = deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_SN(shape=(256, 160, 48, 1), classes=2)
    model.load_weights('/home/zzr/Data/pancreas/script/models/pancreas_weights.160.h5')
    evaluate_list = []
    for img_name in img_names:
        print img_name
        img = nib.load(os.path.join(img_path, img_name))
        mask = nib.load(os.path.join(mask_path, img_name))
        img_data = img.get_data()
        mask_data = mask.get_data()
        x = img_data[np.newaxis, :, :, :, np.newaxis]
        y = mask_data[np.newaxis, :, :, :, np.newaxis]
        evaluate = model.evaluate(x, y, batch_size=1, verbose=1)
        evaluate_list.append(evaluate[1])
        mean = np.mean(evaluate_list)
        std = np.std(evaluate_list)
    print evaluate_list
    print mean, std


def apply_crf(imgvol, segvol, probvol, pred_name):
    print "Running CRF"
    crfparams = {'max_iterations': 5,
                 'dynamic_z': True,
                 'ignore_memory': True,
                 'bilateral_x_std': 2.5318, 'bilateral_y_std': 8.07058, 'bilateral_z_std': 3.29843,
                 'pos_x_std': 0.26038, 'pos_y_std': 1.68586, 'pos_z_std': 0.95493,
                 'bilateral_intensity_std': 6.1507, 'bilateral_w': 38.33788,
                 'pos_w': 6.31136}
    pro = CRFProcessor.CRF3DProcessor(**crfparams)
    print np.max(imgvol), np.min(imgvol)
    liver_pred = pro.set_data_and_run(imgvol, probvol)
    np.save(pred_name, liver_pred)

    _dice = metric.dc(liver_pred == 1, segvol == 1)
    print "Dice before CRF: " + str(_dice)


if __name__ == "__main__":
    img_path = '/media/zzr/Data/Task07_Pancreas/ChangHai/preprocess'
    mask_path = '/home/zzr/Data/pancreas/overall_train_test/test_mask'
    save_path = '/media/zzr/Data/Task07_Pancreas/ChangHai/result'
    test_train(img_path, save_path)
    # evaluate_test(img_path, mask_path)
