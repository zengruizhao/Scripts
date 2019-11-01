# coding=utf-8
import os
import numpy as np

imgPath = '/media/zzr/Data/skin_xml/phase2/semantic/2019-02-2815.44.02.ndpi'
maskPath = '/media/zzr/Data/skin_xml/phase2/semantic/2019-02-2815.44.02.ndpi'
txtPath = '/media/zzr/Data/skin_xml/phase2/semantic'

with open(os.path.join(txtPath, 'train.txt'), 'a') as file:
    for sub in os.listdir(imgPath):
        file.write(os.path.join(imgPath, sub) + ' ' + os.path.join(maskPath, sub) + '\n')

print len(os.listdir(imgPath))
print 'done'