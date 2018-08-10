# coding=utf-8
import numpy as np
import os
import shutil

img_path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/image'
label_path = '/home/zzr/Data/pancreas/caffe_data/256_160_48/label'
img_test_out = '/home/zzr/Data/pancreas/overall_train_test_7_3/test_img'
img_train_out = '/home/zzr/Data/pancreas/overall_train_test_7_3/train_img'
mask_test_out = '/home/zzr/Data/pancreas/overall_train_test_7_3/test_mask'
mask_train_out = '/home/zzr/Data/pancreas/overall_train_test_7_3/train_mask'
for path in [img_test_out, img_train_out, mask_test_out, mask_train_out]:
    if not os.path.exists(path):
        os.makedirs(path)
img_names = os.listdir(img_path)
random_number = img_names
## way 1
np.random.shuffle(random_number)
random_test = random_number[:int(len(img_names)*0.3)]
for img_name in random_test:
    mask_name = os.path.join(label_path, img_name)
    shutil.copyfile(os.path.join(img_path, img_name), os.path.join(img_test_out, img_name))
    shutil.copyfile(mask_name, os.path.join(mask_test_out, img_name))

random_train = random_number[int(len(img_names)*0.3):]
for img_name in random_train:
    mask_name = os.path.join(label_path, img_name)
    shutil.copyfile(os.path.join(img_path, img_name), os.path.join(img_train_out, img_name))
    shutil.copyfile(mask_name, os.path.join(mask_train_out, img_name))
## way 2
# random_number = np.random.permutation(len(img_names))
# random_test = random_number[:int(len(img_names)*0.1)]
# print random_test
# for idx in random_test:
#     img_name = os.path.join(img_path, img_names[idx])
#     mask_name = os.path.join(label_path, img_names[idx])
#     shutil.copyfile(img_name, os.path.join(img_test_out, img_names[idx]))
#     shutil.copyfile(mask_name, os.path.join(mask_test_out, img_names[idx]))
#
# random_train = random_number[int(len(img_names)*0.1):]
# for idx in random_train:
#     img_name = os.path.join(img_path, img_names[idx])
#     mask_name = os.path.join(label_path, img_names[idx])
#     shutil.copyfile(img_name, os.path.join(img_train_out, img_names[idx]))
#     shutil.copyfile(mask_name, os.path.join(mask_train_out, img_names[idx]))

