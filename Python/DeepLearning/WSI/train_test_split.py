# coding=utf-8
"""
    split train and test set randomly
    author: Zengrui Zhao
"""

import numpy as np
import os
import shutil

path = '/media/zzr/Data/skin_xml/cell/data/patch'
result_path = '/media/zzr/Data/skin_xml/cell/data/train_test'
number = 0
for sub_dir in os.listdir(path):
    print sub_dir
    img_all = os.listdir(os.path.join(path, sub_dir))
    rand = np.random.permutation(range(len(img_all)))
    unique = np.unique(rand)
    timer = 0
    for idx in rand:
        img = os.path.join(os.path.join(path, sub_dir), img_all[idx])
        if timer < len(img_all) * 0.2:
            dst_path = os.path.join(result_path+'/test', sub_dir)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
        else:
            dst_path = os.path.join(result_path+'/train', sub_dir)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
        shutil.copyfile(img, os.path.join(dst_path, img_all[idx]))
        timer += 1
        number+=1
print 'success'
print number

