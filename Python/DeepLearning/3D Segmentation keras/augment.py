# coding=utf-8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import config
from PIL import Image


def data_augmentation(img, seg, flip=False, rotate=False):
    """
    :param img:
    :param seg:
    :param flip:
    :param rotate:
    :return: img, seg
    """
    n_dim = len(seg.shape)
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    if flip_axis:
        img = flip_image(img, flip_axis)
        seg = flip_image(seg, flip_axis)
    if rotate:
        angle = np.random.randint(-10, 11)
        img = rotate_image(img, angle)
        seg = rotate_image(seg, angle)
    return img, seg


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if dim != 2:    # dont flip z
            continue
        else:
            if random_boolean():
                axis.append(dim)
    return axis


def random_boolean():
    return np.random.choice([True, False])


def flip_image(image, axis):
    try:
        new_data = np.array(image)
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(np.array(image), axis=axis)
    return new_data


def rotate_image(img, angle):
    outImage = np.copy(img)
    for slice in np.arange(img.shape[2]):
        data = img[..., slice]
        out = Image.fromarray(data.astype('float32'))
        out = out.rotate(angle)
        outImage[..., slice] = np.array(out)

    return outImage


def keras_augmentation(x, y):
    """
    can't be utilized in 3D
    :param x:
    :param y:
    :return:
    """
    data_gen_args = dict(featurewise_center=True,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         zca_whitening=False,
                         zca_epsilon=1e-6,
                         rotation_range=0.,
                         width_shift_range=0.,
                         height_shift_range=0.,
                         shear_range=0.,
                         zoom_range=0.,
                         channel_shift_range=0.,
                         fill_mode='nearest',
                         cval=0.,
                         horizontal_flip=False,
                         vertical_flip=False,
                         rescale=None,
                         preprocessing_function=None,
                         data_format=K.image_data_format())
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    image_datagen.fit(x, augment=True,seed=seed)
    mask_datagen.fit(y, augment=True, seed=seed)
    image_generator = image_datagen.flow(x, batch_size=config["batch_size"], seed=seed)
    mask_generator = mask_datagen.flow(y, batch_size=config["batch_size"], seed=seed)
    return zip(image_generator, mask_generator)
