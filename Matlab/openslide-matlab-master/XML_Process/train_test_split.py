# coding=utf-8
import numpy as np
import os
import shutil

img_path = '/media/zzr/SW/Skin_xml/WSI_new/All_patch'
img_test_out = '/media/zzr/SW/Skin_xml/WSI_new/Train_test/test'
img_train_out = '/media/zzr/SW/Skin_xml/WSI_new/Train_test/train'


def split_train_test():
    for i in xrange(1, 7):
        print i
        for path in [img_test_out, img_train_out]:
            if not os.path.exists(os.path.join(path, str(i))):
                os.makedirs(os.path.join(path, str(i)))
        img_names = os.listdir(os.path.join(img_path, str(i)))
        way1(img_names,
             os.path.join(img_path, str(i)),
             os.path.join(img_test_out, str(i)),
             os.path.join(img_train_out, str(i)))


def way1(img_names, img_path, img_test_path, img_train_path):
    np.random.shuffle(img_names)
    random_test = img_names[:int(len(img_names)*0.1)]
    for img_name in random_test:
        shutil.copyfile(os.path.join(img_path, img_name), os.path.join(img_test_path, img_name))

    random_train = img_names[int(len(img_names)*0.1):]
    for img_name in random_train:
        shutil.copyfile(os.path.join(img_path, img_name), os.path.join(img_train_path, img_name))


if __name__ == '__main__':
    split_train_test()

