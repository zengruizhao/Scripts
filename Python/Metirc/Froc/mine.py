import numpy as np
from froc import computeAndPlotFROC
from skimage.io import imread
import os

pred_path = '/home/zzr/Data/IDRiD/New/OD/Data/seg_result/unet_train/'
label_path='/home/zzr/Data/IDRiD/New/OD/Data/pre_aug/train_mask/'
if __name__ == '__main__':
    
    np.random.seed(1)    
    
    #parameters
    save_path = 'haha.png'
    nbr_of_thresholds = 11
    range_threshold = [0., 1.]
    allowedDistance = 0
    write_thresholds = False
    #
    files = os.listdir(pred_path)
    ind = 0
    ground_truth=[]
    proba_map = []
    for file_ in files:
        seac_name = file_.split('.')[0]+'.png'
        pred = imread(os.path.join(pred_path,seac_name))
        name = file_.split('_seg')[0]+'_OD.png'
        label = imread(os.path.join(label_path,name))
        ground_truth.append(label)
        proba_map.append(pred)

    #read the data
    #ground_truth = np.expand_dims(imageio.imread('/home/zzr/Data/IDRiD/T1/EX/Data/aug/test_mask/IDRiD_03.png'),axis=0)
    #proba_map = np.expand_dims(imageio.imread('/home/zzr/Data/IDRiD/T1/EX/Data/seg_result/segnet/prob2/IDRiD_03_seg.png'),axis=0)
    ground_truth = np.array(ground_truth)
    proba_map = np.array(proba_map)
    print ground_truth.shape
    print proba_map.shape
    #plot FROC
    computeAndPlotFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold, save_path, write_thresholds)
  
