# coding=utf-8
import os
import numpy as np
import shutil

path = '/media/zzr/SW/Skin_xml/WSI_20/test/'
out_path = '/media/zzr/SW/Skin_xml/WSI_20/Train_test/test'


#
def statistics_number():
    number = np.zeros((14, 6))
    for i, name in enumerate(os.listdir(path)):
        print name
        wsi = os.path.join(path, name)
        for j, patch in enumerate(os.listdir(wsi)):
            number[i, j] = len(os.listdir(os.path.join(wsi, patch)))

    print number
    print np.sum(number, 0)


# merge img
def merge_img():
    for i, name in enumerate(os.listdir(path)):
        print i+1, name
        wsi = os.path.join(path, name)
        for idx, patch in enumerate(os.listdir(wsi)):
            if not os.path.exists(os.path.join(out_path, str(idx))):
                os.makedirs(os.path.join(out_path, str(idx)))
            for img in os.listdir(os.path.join(wsi, patch)):
                shutil.copyfile(os.path.join(os.path.join(wsi, patch), img),
                                os.path.join(os.path.join(out_path, str(idx)), img))


if __name__ == '__main__':
    # statistics_number()
    merge_img()