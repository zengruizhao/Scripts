# coding=utf-8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import config
from PIL import Image
import cv2


def data_augmentation(img, seg, flip=False, rotate=False, affine=False):
    """
    :param img:
    :param seg:
    :param flip:
    :param rotate:
    :param affine
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
        angle = np.random.randint(-5, 6)
        img = rotate_image(img, angle)
        seg = rotate_image(img, angle)
    if affine:
        if random_boolean():
            img, seg = affine_transformation(img, img, img.shape[1] * 0.03, random_state=None)

    return img, seg


def affine_transformation(image, mask, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    channel = image.shape[-1]
    image = np.concatenate((image, mask), axis=-1)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_NEAREST)

    return image[..., 0:channel], image[..., channel:]


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
