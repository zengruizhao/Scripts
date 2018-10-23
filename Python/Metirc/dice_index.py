#-*- coding: utf-8
from medpy import metric
from skimage.io import imread
import os
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy

#pred_path = '/home/mw/mw/caffe_segnet/data/seg_dierci/'
pred_path = '/home/zzr/Data/ISIC/train_test/aug/test_result_original/'
#CV_path = '/media/mw/mw_research/data/ruxian/120/120_zhong_no_zhong/C-V'
#SEAC_path = '/media/mw/mw_research/data/code/matlab/SEAC/results'
#fcm_path = '/media/mw/mw_research/data/ruxian/120/120_zhong_no_zhong/after_change_144_144/fcm_results'
#pred_path ='/media/mw/mw_research/data/ruxian/120/144_144_seg/test/CV_seg'
#label_path = '/media/mw/mw_research/data/ruxian/120/144_144_seg/test/144_144_mask_test_vis'
label_path='/home/zzr/Data/ISIC/train_test/aug/test_mask/'
def Jc(input1, input2):
    """
    Jaccard coefficient
    
    Computes the Jaccard coefficient between the binary objects in two images.
    
    Parameters
    ----------
    input1: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    input2: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.

    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `input1` and the
        object(s) in `input2`. It ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Notes
    -----
    This is a real metric.
    """
    input1 = numpy.atleast_1d(input1.astype(numpy.bool))
    input2 = numpy.atleast_1d(input2.astype(numpy.bool))
    
    intersection = numpy.count_nonzero(input1 & input2)
    union = numpy.count_nonzero(input1 | input2)
    
    jc = float(intersection) / float(union)
    
    return jc
def Dice(input1, input2):
	input1 = numpy.atleast_1d(input1.astype(numpy.bool))
	input2 = numpy.atleast_1d(input2.astype(numpy.bool))
    
	intersection = numpy.count_nonzero(input1 & input2)
	union = numpy.count_nonzero(input1 | input2) + intersection
    
	dice = 2 * float(intersection) / float(union)
    
	return dice
dice_indexs = 0
num = 0
files = os.listdir(pred_path)
for file_ in files:
    seac_name = file_.split('.')[0]+'.png'
    
    pred = imread(os.path.join(pred_path,seac_name))#针对shape（144,144,3）
    #print(pred_list)
    name = file_.split('_seg')[0]+'.png'
    #name = file_
    label = imread(os.path.join(label_path,name))
    #dice_index = metric.precision(pred,label)
    #dice_index = metric.recall(pred,label)
    #dice_index = metric.dc(pred,label)
    #dice_index = Jc(pred,label)
    dice_index = Jc(pred,label)
    if dice_index==0:
        #print(file_)
        continue
    # print(dice_index)
    # print(name)
    dice_indexs += dice_index
    num +=1
print(num)
mean_dice = dice_indexs/num
print(mean_dice)

