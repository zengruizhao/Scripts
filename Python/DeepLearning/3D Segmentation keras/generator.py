import numpy as np
import os
import nibabel as nib
from config import config
from utils import split_filename, extract_precess_mode
from augment import data_augmentation
from keras import backend as K
K.set_image_dim_ordering('tf')


def get_training_and_testing_generators(train_data_path, train_seg_path, batch_size, data_split):
    train_data_name = os.listdir(train_data_path)
    training_generator = data_generator(train_data_path,
                                        train_seg_path,
                                        train_data_name,
                                        batch_size=batch_size,
                                        flip=True,
                                        rotate=True)
    # Set the number of training and testing samples per epoch correctly
    nb_training_samples = len(train_data_name)
    test_img_path = '/home/zzr/Data/pancreas/overall_train_test_7_3/test_img'
    test_seg_path = '/home/zzr/Data/pancreas/overall_train_test_7_3/test_mask'
    testing_generator = data_generator(test_img_path,
                                       test_seg_path,
                                       os.listdir(test_img_path),
                                       batch_size=batch_size,
                                       flip=False,
                                       rotate=False)
    nb_testing_samples = len(os.listdir(test_img_path))
    return training_generator, nb_training_samples//batch_size, testing_generator, nb_testing_samples//batch_size


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        np.random.shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(train_data_path, train_seg_path, data_name, batch_size=1, flip=False, rotate=False):
    while True:
        np.random.shuffle(data_name)
        x_list = list()
        y_list = list()
        for index in data_name:
            img_ = nib.load(os.path.join(train_data_path, index))
            img = img_.get_data()

            seg_ = nib.load(os.path.join(train_seg_path, index))
            seg = seg_.get_data()
            try:
                img_aug, seg_aug = data_augmentation(img, seg, flip=flip, rotate=rotate)
                assert len(np.unique(seg_aug)) == 2, 'wrong label'
                img, seg = img_aug, seg_aug
            except AssertionError:
                pass
            x_list.append(img)
            y_list.append(seg)
            if len(x_list) == batch_size:
                yield convert_data(x_list, y_list)  # generator
                x_list = list()
                y_list = list()


def convert_data(x_list, y_list):
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    if False:
        label_list = []
        for sample_num in range(y_list.shape[0]):
            samples_list = []
            for label in range(config["n_labels"]):
                temparray = np.zeros(y_list.shape[1:])
                temparray[y_list[sample_num] == label] = 1
                samples_list.append(temparray)
            label_list.append(samples_list)
        y_list = np.rollaxis(np.array(label_list), 1, 5)
    y_list = np.expand_dims(y_list, -1)
    x_list = np.expand_dims(x_list, -1)
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    # return keras_augmentation(x, y)
    return x, y