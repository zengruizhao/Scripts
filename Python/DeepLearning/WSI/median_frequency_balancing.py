# coding=utf-8
"""
    median frequency balancing
    author: Zengrui Zhao
"""

import numpy as np
import os

path = '/home/zzr/Data/ISIC/densenet/train_test/train_aug'
temp = []
Label = []
for label in os.listdir(path):
    Label.append(label)
    temp.append(len(os.listdir(os.path.join(path, label))))
frequency = np.divide(temp, float(np.sum(temp)))
# median = np.median(frequency)
max = np.max(frequency)
weight = np.divide(max, frequency)
weight_2 = weight**2
print weight
print weight_2
print Label