# coding=utf-8
"""
author: Zengrui Zhao
"""
import os
import shutil

path = '/media/zzr/Data/skin_xml/temp/5'
for img in os.listdir(path):
    # os.rename(os.path.join(path, img), os.path.join('/media/zzr/Data/skin_xml/temp/5', '161409_' + img))
    shutil.copyfile(os.path.join(path, img), os.path.join('/home/zzr/Data/Skin/data/All_2/5', img))
