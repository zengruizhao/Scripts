import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, draw
from scipy import misc
import cv2

def split_filename(filename):
    return filename.split('.')[0].split('-')[1]

def extract_precess_mode(name):
    return name.split('_')[0]


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def get_class_weight():
    seg_path = '/home/mi/smj/data/liver/new_unet_seg/get128shape_moredata_addnorm_reflect/seg'
    seg_names = os.listdir(seg_path)

    final_num = []
    label = [0, 1]
    for seg_name in seg_names:
        print seg_name
        temp=[]
        seg_data = np.load(os.path.join(seg_path, seg_name), mmap_mode='r')
        # seg_data = misc.imread('')
        for i in range(len(label)):
            temp_label = label[i]
            label_num = np.array(seg_data == temp_label).astype(np.uint8).sum()
            temp.append(label_num)
        print temp
        final_num.append(temp)
        # print final_num
    final_num = np.array(final_num)
    sum_num = final_num.sum(axis=0)
    print '*' * 70
    print 'sum_num'
    print sum_num
    median=sum(sum_num) / len(sum_num)
    class_weighting = median * 1. / sum_num
    print '*' * 70
    print 'class_weighting'
    print class_weighting


def test_contour():
    # img = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png/img/volume-100.npy/volume-258_100.png')
    # img_contour = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png/pred/volume-100.npy/pred-258_100.png')
    # img_contour = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/pred/volume-100.npy/pred-258_100.png')
    # img_contour = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/seg/volume-100.npy/segmentation-258_100.png')

    # img = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/img/volume-100.npy/volume-258_100.png')
    # img_contour = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/seg/volume-100.npy/segmentation-258_100.png')
    # pred_contour = misc.imread('/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/pred/volume-100.npy/pred-258_100.png')

    # img = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_img/volume-100.npy/volume-261_100.png')
    # img_contour = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_seg/volume-100.npy/segmentation-261_100.png')
    # pred_contour = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_pred/volume-100.npy/pred-261_100.png')

    # img = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_img/volume-100.npy/volume-261_100.png')
    # img_contour = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_seg/volume-100.npy/segmentation-261_100.png')
    # pred_contour = misc.imread(
    #     '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/noaugment_170807_moredata_laststd/my100png_bilinear/y_pred/volume-100.npy/pred-261_100.png')

    img = misc.imread(
        '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/BN_RELU/pred_400/png_94_400/z_img/volume-94.npy/volume-111_94.png')
    img_contour = misc.imread(
        '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/BN_RELU/pred_400/png_94_400/z_seg/volume-94.npy/segmentation-111_94.png')
    pred_contour = misc.imread(
        '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/BN_RELU/pred_400/png_94_400/z_pred/volume-94.npy/pred-111_94.png')

    img_contour = np.array(img_contour >0.5).astype(np.uint8)
    pred_contour = np.array(pred_contour > 0.5).astype(np.uint8)

    # print pred_contour.shape

    contours = measure.find_contours(img_contour, 0.5)
    pred_contours = measure.find_contours(pred_contour, 0.5)

    print len(pred_contours)
    print len(contours)

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 16))
    # ax0.imshow(img, plt.cm.gray)
    ax1.imshow(img, plt.cm.gray)
    # for n, contour in enumerate(contours):
    #     ax1.plot(contour[:, 1], contour[:, 0],'g', linewidth=2)

    for contour, pred_ in zip(contours, pred_contours):
        # print contour
        # print pred_
        ax1.plot(contour[:, 1], contour[:, 0],'#00FF00', linewidth=8)
        # ax1.plot(pred_[:, 1], pred_[:, 0], 'r', linewidth=4)

    for pred_ in pred_contours:
        for pred__ in [pred_]:
            # print len(pred__)
            ax1.plot(pred__[:, 1], pred__[:, 0], 'r', linewidth=8)

    ax1.axis('image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # plt.savefig("./test_100_258_bilinear_crf.png")
    plt.savefig("./z_111_bn.png")
    plt.show()

def my_cv_test():
    img_path = '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/new1_400/png_400/x_img/volume-94.npy/volume-124_94.png'
    img_contour_path = '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/new1_400/png_400/x_seg/volume-94.npy/segmentation-124_94.png'
    pred_contour_path = '/home/mi/smj/data/liver/new_unet_seg/moredata_addnorm_process_std/get128shape_moredata_laststd_test/new1_400/png_400/x_pred/volume-94.npy/pred-124_94.png'

    img = cv2.imread(img_path)
    seg = cv2.imread(img_contour_path)
    pred = cv2.imread(pred_contour_path)

    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(pred, 127, 255, 0)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContour(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('img', img)
    cv2.waitKey()

if __name__=='__main__':
    test_contour()
    # my_cv_test()