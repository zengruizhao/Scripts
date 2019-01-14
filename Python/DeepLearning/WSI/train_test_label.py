# coding=utf-8
"""
    generate the label file for train and test set using caffe
    author: Zengrui Zhao
"""
import numpy as np
import os

path = '/media/zzr/Data/skin_xml/cell/data/augpatch'
for sub in os.listdir(path):
    with open('/media/zzr/Data/skin_xml/cell/data/augpatch/' + str(sub) + '.txt', 'a') as file:
        label_all = os.listdir(os.path.join(path, sub))
        for label in label_all:
            img_all = os.listdir(os.path.join(os.path.join(path, sub), label))
            for img in img_all:
                out_str = label + '/' + img + ' ' + label + '\n'
                file.write(out_str)
    print sub, 'done'
