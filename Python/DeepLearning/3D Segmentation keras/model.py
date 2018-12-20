from keras.engine import Input, Model
from keras.layers import Conv2D, Conv3D, MaxPooling3D, UpSampling3D, Softmax, BatchNormalization, \
    Conv3DTranspose, GlobalAveragePooling3D, GlobalMaxPooling3D, Cropping3D, Dense
from keras.layers.core import Activation, Dropout, RepeatVector, Lambda
from keras.optimizers import Adam, Adadelta, SGD
from config import config
from keras import backend as K
from keras.regularizers import l2
from keras.layers import add, multiply, Dense, LSTM, maximum, average, Reshape
from coord import CoordinateChannel3D
from group_norm import GroupNormalization
from SwitchableNormalization import SwitchableNormalization
import numpy as np
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers.convolutional import ZeroPadding3D
import tensorflow as tf

l2_ratio = 1e-4


def unet_model_3d(shape=None, classes=2):
    inputs = Input(shape)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:",pool1.shape

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    print "conv4 shape:", conv4.shape
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    print "pool4 shape:", pool4.shape

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    print "conv5 shape:", conv5.shape
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    print "conv5 shape:", conv5.shape

    # up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    # up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    # up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    # up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv9)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)
    # act = Activation('sigmoid')(conv10)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adadelta(), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def new_unet_model_3d(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                      initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv7)

    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)


def deconv_conv_unet_model_3d_conv_add_48_dalitaion(shape, classes=2):
    inputs = Input(shape)
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv4)
    print "deconv6 shape:", deconv6.shape
    deconv6 = BatchNormalization()(deconv6)
    deconv6 = Conv3D(128, 3, activation='relu', padding='same')(deconv6)
    deconv6 = BatchNormalization()(deconv6)
    conv_conv4 = Conv3D(128, 3, activation='relu', padding='same')(conv3)
    conv_conv4 = BatchNormalization()(conv_conv4)
    up6 = concatenate([deconv6, conv_conv4],axis=-1)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv6)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = Conv3D(64, 3, activation='relu', padding='same')(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    conv_conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv_conv3 = BatchNormalization()(conv_conv3)
    up7 = concatenate([deconv7, conv_conv3],axis=-1)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',)(conv7)
    deconv8 = BatchNormalization()(deconv8)
    deconv8 = Conv3D(32, 3, activation='relu', padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    conv_conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv_conv2 = BatchNormalization()(conv_conv2)
    up8 = concatenate([deconv8, conv_conv2], axis=-1)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = BatchNormalization()(conv10)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)
    # act = Activation('sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_GN(shape, classes=2): # Good
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x)
    conv1 = GroupNormalization(groups=32)(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = GroupNormalization(groups=32)(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:",pool1.shape

    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = GroupNormalization(groups=32)(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(groups=32)(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = GroupNormalization(groups=32)(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(groups=32)(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool3)
    conv4 = GroupNormalization(groups=32)(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv4)
    conv4 = GroupNormalization(groups=32)(conv4)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(128, 2,
                         strides=(2, 2, 2),
                         padding='valid')(conv4)
    print "deconv6 shape:", deconv6.shape
    deconv6 = GroupNormalization(groups=32)(deconv6)
    deconv6 = Conv3D(128, 3, activation='relu', padding='same')(deconv6)
    deconv6 = GroupNormalization(groups=32)(deconv6)
    conv_conv4 = Conv3D(128, 3, activation='relu', padding='same')(conv3)
    conv_conv4 = GroupNormalization(groups=32)(conv_conv4)
    up6 = concatenate([deconv6, conv_conv4],axis=-1)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = GroupNormalization(groups=32)(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = GroupNormalization(groups=32)(conv6)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv6)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = Conv3D(64, 3, activation='relu', padding='same')(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    conv_conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv_conv3 = BatchNormalization()(conv_conv3)
    up7 = concatenate([deconv7, conv_conv3],axis=-1)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)


    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',)(conv7)
    deconv8 = BatchNormalization()(deconv8)
    deconv8 = Conv3D(32, 3, activation='relu', padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    conv_conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv_conv2 = BatchNormalization()(conv_conv2)
    up8 = concatenate([deconv8, conv_conv2], axis=-1)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)


    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = BatchNormalization()(conv10)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)
    # act = Activation('sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_coordconv_gn_modified(shape, classes=2):
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_without_dilation(x, filters=32)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_relu_gn_without_dilation(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_without_dilation(pool1, filters=64)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_relu_gn_without_dilation(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_without_dilation(pool2, filters=128)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_relu_gn_without_dilation(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    pool3 = Dropout(rate=0.5)(pool3)

    conv4 = conv3d_relu_gn_without_dilation(pool3, filters=256)
    print "conv4 shape:", conv4.shape
    conv4 = conv3d_relu_gn_without_dilation(conv4, filters=256)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv4)
    deconv6 = GroupNormalization(groups=32)(deconv6)
    
    up6 = concatenate([deconv6, conv3], axis=-1)
    conv6 = conv3d_relu_gn_without_dilation(up6, filters=128)
    conv6 = conv3d_relu_gn_without_dilation(conv6, filters=128)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=32)(deconv7)

    up7 = concatenate([deconv7, conv2], axis=-1)
    conv7 = conv3d_relu_gn_without_dilation(up7, filters=64)
    conv7 = conv3d_relu_gn_without_dilation(conv7, filters=64)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=32)(deconv8)

    up8 = concatenate([deconv8, conv1], axis=-1)
    conv8 = conv3d_relu_gn_without_dilation(up8, filters=32)
    conv8 = conv3d_relu_gn_without_dilation(conv8, filters=32)

    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = GroupNormalization(groups=2)(conv10)

    act = Conv3D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deeplab_aunet(shape, classes=2):
    inputs = Input(shape)
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape
    ##
    conv = Conv3D(128, (1, 1, 1), activation='relu', padding='same')(pool3)
    conv = BatchNormalization()(conv)

    aspp_conv1 = Conv3D(128, (1, 1, 1), dilation_rate=(1, 1, 1), activation='relu', padding='SAME')(conv)
    aspp_conv1 = BatchNormalization()(aspp_conv1)
    aspp_conv2 = Conv3D(128, (3, 3, 3), dilation_rate=(1, 1, 1), activation='relu', padding='SAME')(conv)
    aspp_conv2 = BatchNormalization()(aspp_conv2)
    aspp_conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='SAME')(conv)
    aspp_conv3 = BatchNormalization()(aspp_conv3)
    aspp_conv4 = Conv3D(128, (3, 3, 3), dilation_rate=(5, 5, 5), activation='relu', padding='SAME')(conv)
    aspp_conv4 = BatchNormalization()(aspp_conv4)
    aspp_conv5 = Conv3D(128, (3, 3, 3), dilation_rate=(8, 8, 8), activation='relu', padding='SAME')(conv)
    aspp_conv5 = BatchNormalization()(aspp_conv5)

    aspp_pool = GlobalAveragePooling3D()(conv)

    concat = concatenate([aspp_conv1, aspp_conv2, aspp_conv3, aspp_conv4, aspp_conv5], axis=-1)
    concat_conv = Conv3D(128, (1, 1, 1), activation='relu', padding='SAME')(concat)
    concat_conv = BatchNormalization()(concat_conv)
    concat_global_pool = multiply([aspp_pool, concat_conv])
    # concat_global_pool = aspp_pool * concat_conv
    print "concat_global_pool shape:", concat_global_pool.shape
    final_aspp = add([conv, concat_global_pool])
    print "final_aspp shape:", final_aspp.shape
    ##
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(final_aspp)
    conv4 = BatchNormalization()(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    print "conv4 shape:", conv4.shape
    concat1 = concatenate([conv4, pool3], axis=-1)
    concat1 = Conv3D(128, (1, 1, 1), padding='same')(concat1)
    #####################################################
    deconv1 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid')(concat1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    deconv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(deconv1)
    deconv1 = BatchNormalization()(deconv1)

    concat2 = concatenate([deconv1, pool2], axis=-1)
    concat2 = Conv3D(64, (1, 1, 1), padding='same')(concat2)
    deconv2 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid')(concat2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Conv3D(64, 3, activation='relu', padding='same')(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    deconv2 = Conv3D(32, 3, activation='relu', padding='same')(deconv2)
    deconv2 = BatchNormalization()(deconv2)

    concat3 = concatenate([deconv2, pool1], axis=-1)
    concat3 = Conv3D(32, (1, 1, 1), padding='same')(concat3)
    deconv3 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid', )(concat3)
    deconv3 = BatchNormalization()(deconv3)
    deconv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(deconv3)
    deconv3 = BatchNormalization()(deconv3)

    act = Conv3D(classes, (1, 1, 1), activation='softmax')(deconv3)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_GN_best(shape, classes=2): # Need more memory
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = Conv3D(16, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x)
    conv1 = GroupNormalization(groups=16)(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(16, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = GroupNormalization(groups=16)(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = GroupNormalization(groups=32)(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = GroupNormalization(groups=32)(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = GroupNormalization(groups=64)(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = GroupNormalization(groups=64)(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool3)
    conv4 = GroupNormalization(groups=128)(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv4)
    conv4 = GroupNormalization(groups=128)(conv4)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv4)
    print "deconv6 shape:", deconv6.shape
    deconv6 = GroupNormalization(groups=64)(deconv6)
    deconv6 = Conv3D(64, 3, activation='relu', padding='same')(deconv6)
    deconv6 = GroupNormalization(groups=64)(deconv6)
    conv_conv4 = Conv3D(64, 3, activation='relu', padding='same')(conv3)
    conv_conv4 = GroupNormalization(groups=64)(conv_conv4)
    up6 = concatenate([deconv6, conv_conv4],axis=-1)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = GroupNormalization(groups=64)(conv6)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = GroupNormalization(groups=64)(conv6)

    deconv7 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv6)
    deconv7 = GroupNormalization(groups=32)(deconv7)
    deconv7 = Conv3D(32, 3, activation='relu', padding='same')(deconv7)
    deconv7 = GroupNormalization(groups=32)(deconv7)
    conv_conv3 = Conv3D(32, 3, activation='relu', padding='same')(conv2)
    conv_conv3 = GroupNormalization(groups=32)(conv_conv3)
    up7 = concatenate([deconv7, conv_conv3],axis=-1)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = GroupNormalization(groups=32)(conv7)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = GroupNormalization(groups=32)(conv7)

    deconv8 = Conv3DTranspose(16, 2,
                              strides=(2, 2, 2),
                              padding='valid',)(conv7)
    deconv8 = GroupNormalization(groups=16)(deconv8)
    deconv8 = Conv3D(16, 3, activation='relu', padding='same')(deconv8)
    deconv8 = GroupNormalization(groups=16)(deconv8)
    conv_conv2 = Conv3D(16, 3, activation='relu', padding='same')(conv1)
    conv_conv2 = GroupNormalization(groups=16)(conv_conv2)
    up8 = concatenate([deconv8, conv_conv2], axis=-1)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = GroupNormalization(groups=16)(conv8)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = GroupNormalization(groups=16)(conv8)

    act = Conv3D(classes, 1, activation='softmax')(conv8)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_SN(shape, classes=2):  # reduced feature map
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = Conv3D(16, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x)
    conv1 = SwitchableNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(16, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = SwitchableNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = SwitchableNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = SwitchableNormalization()(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = SwitchableNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = SwitchableNormalization()(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool3)
    conv4 = SwitchableNormalization()(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv4)
    conv4 = SwitchableNormalization()(conv4)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv4)
    print "deconv6 shape:", deconv6.shape
    deconv6 = SwitchableNormalization()(deconv6)

    up6 = concatenate([deconv6, conv3], axis=-1)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = SwitchableNormalization()(conv6)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = SwitchableNormalization()(conv6)

    deconv7 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid')(conv6)
    deconv7 = SwitchableNormalization()(deconv7)

    up7 = concatenate([deconv7, conv2], axis=-1)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = SwitchableNormalization()(conv7)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = SwitchableNormalization()(conv7)

    deconv8 = Conv3DTranspose(16, 2,
                              strides=(2, 2, 2),
                              padding='valid', )(conv7)
    deconv8 = SwitchableNormalization()(deconv8)

    up8 = concatenate([deconv8, conv1], axis=-1)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = SwitchableNormalization()(conv8)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = SwitchableNormalization()(conv8)

    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = SwitchableNormalization()(conv10)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_coordconv_gn_modified1(shape, classes=2):  # ASPP + Attention
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(x, filters=32)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_relu_gn_dilation(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = Dropout(rate=0.5)(pool1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_dilation(pool1, filters=64)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_relu_gn_dilation(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = Dropout(rate=0.5)(pool2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_dilation(pool2, filters=128)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_relu_gn_dilation(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    # pool3 = Dropout(rate=0.5)(pool3)
    conv4 = conv3d_relu_gn_dilation(pool3, filters=256)
    conv4 = convolutional_block_attention_module(conv4, out_dim=256)
    print "conv4 shape:", conv4.shape
    conv4 = conv3d_relu_gn_dilation(conv4, filters=256)
    conv4 = convolutional_block_attention_module(conv4, out_dim=256)
    print "conv4 shape:", conv4.shape

    conv = conv3d_relu_gn_dilation(conv4, filters=256, kernel_size=(1, 1, 1))

    aspp_conv1 = conv3d_relu_gn_dilation(conv, filters=256, kernel_size=(1, 1, 1))
    aspp_conv2 = conv3d_relu_gn_dilation(conv, filters=256, kernel_size=(3, 3, 3))
    aspp_conv3 = conv3d_relu_gn_dilation(conv, filters=256, kernel_size=(3, 3, 3), dilation_rate=(2, 2, 2))
    aspp_conv4 = conv3d_relu_gn_dilation(conv, filters=256, kernel_size=(3, 3, 3), dilation_rate=(5, 5, 5))
    aspp_conv5 = conv3d_relu_gn_dilation(conv, filters=256, kernel_size=(3, 3, 3), dilation_rate=(8, 8, 8))

    aspp_pool = GlobalAveragePooling3D()(conv)

    concat = concatenate([aspp_conv1, aspp_conv2, aspp_conv3, aspp_conv4, aspp_conv5], axis=-1)
    concat_conv = conv3d_relu_gn_dilation(concat, filters=256, kernel_size=(1, 1, 1))
    concat_global_pool = multiply([aspp_pool, concat_conv])

    print "concat_global_pool shape:", concat_global_pool.shape
    final_aspp = add([conv, concat_global_pool])
    print "final_aspp shape:", final_aspp.shape

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(final_aspp)
    deconv6 = GroupNormalization(groups=128)(deconv6)

    up6 = concatenate([deconv6, conv3], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up6, filters=128)
    conv6 = conv3d_relu_gn_dilation(conv6, filters=128)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=64)(deconv7)

    up7 = concatenate([deconv7, conv2], axis=-1)
    conv7 = conv3d_relu_gn_dilation(up7, filters=64)
    conv7 = conv3d_relu_gn_dilation(conv7, filters=64)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=32)(deconv8)

    up8 = concatenate([deconv8, conv1], axis=-1)
    conv8 = conv3d_relu_gn_dilation(up8, filters=32)
    conv8 = conv3d_relu_gn_dilation(conv8, filters=32)

    act = Conv3D(classes, 1, activation='softmax')(conv8)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def Vnet_modified(shape, classes=2):
    inputs = Input(shape)
    # inputs = CoordinateChannel3D()(inputs)
    conv1 = Conv3D(filters=16, kernel_size=3, padding='same')(inputs)
    add_conv1 = add([inputs, conv1])
    add_conv1 = ReLU()(add_conv1)
    add_conv1 = BatchNormalization()(add_conv1)
    down1 = conv3d_prelu(add_conv1, filters=32, kernel_size=2, padding='valid', strides=2)
    down1 = Dropout(0.9)(down1)
    # down1 = MaxPooling3D()(add_conv1)
    # down1 = conv3d_prelu(down1, filters=16, kernel_size=1, padding='same')
    # down1 = conv3d_prelu(down1, filters=32, kernel_size=3, padding='same')
    print 'down1 shape:', down1.shape

    conv2 = Conv3D(filters=32, kernel_size=3, padding='same')(down1)
    add_conv2 = add([down1, conv2])
    add_conv2 = ReLU()(add_conv2)
    add_conv2 = BatchNormalization()(add_conv2)
    down2 = conv3d_prelu(add_conv2, filters=64, kernel_size=(2, 2, 2), padding='valid', strides=2)
    down2 = Dropout(0.9)(down2)
    # down2 = MaxPooling3D()(add_conv2)
    # down2 = conv3d_prelu(down2, filters=32, kernel_size=1, padding='same')
    # down2 = conv3d_prelu(down2, filters=64, kernel_size=3, padding='same')
    print 'down2 shape:', down2.shape

    conv3 = conv3d_prelu(down2, filters=32)
    conv3 = Conv3D(filters=64, kernel_size=3, padding='same')(conv3)
    add_conv3 = add([down2, conv3])
    add_conv3 = ReLU()(add_conv3)
    add_conv3 = BatchNormalization()(add_conv3)
    down3 = conv3d_prelu(add_conv3, filters=128, kernel_size=(2, 2, 2), padding='valid', strides=2)
    down3 = Dropout(0.9)(down3)
    # down3 = MaxPooling3D()(add_conv3)
    # down3 = conv3d_prelu(down3, filters=64, kernel_size=1, padding='same')
    # down3 = conv3d_prelu(down3, filters=128, kernel_size=3, padding='same')
    # down3 = Dropout(0.5)(down3)
    print 'down3 shape:', down3.shape

    conv4 = conv3d_prelu(down3, filters=128)
    conv4 = conv3d_prelu(conv4, filters=128)
    conv4 = Conv3D(filters=128, kernel_size=3, padding='same')(conv4)
    add_conv4 = add([down3, conv4])
    add_conv4 = ReLU()(add_conv4)
    add_conv4 = BatchNormalization()(add_conv4)
    deconv1 = Conv3DTranspose(filters=64, kernel_size=2, strides=2)(add_conv4)
    deconv1 = ReLU()(deconv1)
    deconv1 = BatchNormalization()(deconv1)
    print 'deconv1 shape', deconv1.shape

    concatenate1 = concatenate([deconv1, add_conv3])
    concatenate1_1 = conv3d_prelu(concatenate1, filters=128)
    concatenate1_1 = Conv3D(filters=128, kernel_size=3, padding='same')(concatenate1_1)
    add_deconv1 = add([concatenate1_1, concatenate1])
    add_deconv1 = ReLU()(add_deconv1)
    add_deconv1 = BatchNormalization()(add_deconv1)

    deconv2 = Conv3DTranspose(filters=32, kernel_size=2, strides=2)(add_deconv1)
    deconv2 = ReLU()(deconv2)
    deconv2 = BatchNormalization()(deconv2)
    print 'deconv2 shape', deconv2.shape

    concatenate2 = concatenate([deconv2, add_conv2])
    concatenate2_1 = Conv3D(filters=64, kernel_size=3, padding='same')(concatenate2)
    add_deconv2 = add([concatenate2_1, concatenate2])
    add_deconv2 = ReLU()(add_deconv2)
    add_deconv2 = BatchNormalization()(add_deconv2)

    deconv3 = Conv3DTranspose(filters=16, kernel_size=2, strides=2)(add_deconv2)
    deconv3 = ReLU()(deconv3)
    deconv3 = BatchNormalization()(deconv3)
    print 'deconv3 shape', deconv3.shape

    concatenate3 = concatenate([deconv3, add_conv1])
    concatenate3_1 = Conv3D(filters=32, kernel_size=3, padding='same',)(concatenate3)
    add_deconv3 = add([concatenate3_1, concatenate3])
    add_deconv3 = ReLU()(add_deconv3)
    add_deconv3 = BatchNormalization()(add_deconv3)

    prediction = Conv3D(filters=classes, kernel_size=1, padding='same', activation='softmax')(add_deconv3)
    model = Model(inputs=inputs, outputs=prediction)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_coordconv_gn_deep(shape, classes=2):
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(x, filters=16)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_relu_gn_dilation(conv1, filters=16)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = Dropout(rate=0.5)(pool1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_dilation(pool1, filters=32)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_relu_gn_dilation(conv2, filters=32)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = Dropout(rate=0.5)(pool2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_dilation(pool2, filters=64)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_relu_gn_dilation(conv3, filters=64)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    # pool3 = Dropout(rate=0.5)(pool3)
    print "pool3 shape:", pool3.shape

    conv4 = conv3d_relu_gn_dilation(pool3, filters=128)
    print "conv4 shape:", conv4.shape
    conv4 = conv3d_relu_gn_dilation(conv4, filters=128)
    print "conv4 shape:", conv4.shape
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    print 'pool4 shape', pool4.shape

    conv5 = conv3d_relu_gn_dilation(pool4, filters=256)
    conv5 = conv3d_relu_gn_dilation(conv5, filters=256)

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv6 = GroupNormalization(groups=32)(deconv6)

    up6 = concatenate([deconv6, conv4], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up6, filters=128)
    conv6 = conv3d_relu_gn_dilation(conv6, filters=128)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=32)(deconv7)

    up7 = concatenate([deconv7, conv3], axis=-1)
    conv7 = conv3d_relu_gn_dilation(up7, filters=64)
    conv7 = conv3d_relu_gn_dilation(conv7, filters=64)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=16)(deconv8)

    up8 = concatenate([deconv8, conv2], axis=-1)
    conv8 = conv3d_relu_gn_dilation(up8, filters=32)
    conv8 = conv3d_relu_gn_dilation(conv8, filters=32)

    deconv9 = Conv3DTranspose(16, 2, strides=2, padding='valid', activation='relu')(conv8)
    deconv9 = GroupNormalization(groups=16)(deconv9)

    up9 = concatenate([deconv9, conv1], axis=-1)
    conv9 = conv3d_relu_gn_dilation(up9, filters=16)
    conv9 = conv3d_relu_gn_dilation(conv9, filters=16)

    act1 = Conv3D(classes, 1, activation='softmax', name='JACCARD')(conv9)
    conv10 = conv3d_relu_gn_dilation(conv9, filters=2, kernel_size=1)
    act2 = Conv3D(1, 1, activation='sigmoid', name='DICE_Smooth')(conv10)

    model = Model(inputs=inputs, outputs=[act1, act2])

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=[jaccard_loss, dice_coef_loss],
                  metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_coordconv_gn_modified_resnet(shape, classes=2):
    """
    model is too big and the speed is too slow
    :param shape:
    :param classes:
    :return:
    """
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(x, filters=32)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_relu_gn_dilation(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = Dropout(rate=0.5)(pool1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_dilation(pool1, filters=64)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_relu_gn_dilation(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = Dropout(rate=0.5)(pool2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_dilation(pool2, filters=128)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_relu_gn_dilation(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    # pool3 = Dropout(rate=0.5)(pool3)
    conv4 = conv3d_relu_gn_dilation(pool3, filters=256, kernel_size=(1, 1, 1))
    res = res_block(conv4, number=7, filters=256)

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(res)
    deconv6 = GroupNormalization(groups=128)(deconv6)

    up6 = concatenate([deconv6, conv3], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up6, filters=128)
    conv6 = conv3d_relu_gn_dilation(conv6, filters=128)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=64)(deconv7)

    up7 = concatenate([deconv7, conv2], axis=-1)
    conv7 = conv3d_relu_gn_dilation(up7, filters=64)
    conv7 = conv3d_relu_gn_dilation(conv7, filters=64)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=32)(deconv8)

    up8 = concatenate([deconv8, conv1], axis=-1)
    conv8 = conv3d_relu_gn_dilation(up8, filters=32)
    conv8 = conv3d_relu_gn_dilation(conv8, filters=32)

    act = Conv3D(classes, 1, activation='softmax')(conv8)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_coordconv_gn_deep_supervision(shape, classes=2):  # modifying...
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(x, filters=16)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_relu_gn_dilation(conv1, filters=16)
    print "conv1 shape:", conv1.shape
    dsn1 = dsn_block(conv1, ks_stride=1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = Dropout(rate=0.5)(pool1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_dilation(pool1, filters=32)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_relu_gn_dilation(conv2, filters=32)
    print "conv2 shape:", conv2.shape
    dsn2 = dsn_block(conv2, ks_stride=2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = Dropout(rate=0.5)(pool2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_dilation(pool2, filters=64)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_relu_gn_dilation(conv3, filters=64)
    print "conv3 shape:", conv3.shape
    dsn3 = dsn_block(conv3, ks_stride=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    # pool3 = Dropout(rate=0.5)(pool3)
    print "pool3 shape:", pool3.shape

    conv4 = conv3d_relu_gn_dilation(pool3, filters=128)
    print "conv4 shape:", conv4.shape
    conv4 = conv3d_relu_gn_dilation(conv4, filters=128)
    print "conv4 shape:", conv4.shape
    dsn4 = dsn_block(conv4, ks_stride=8)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    print 'pool4 shape', pool4.shape

    conv5 = conv3d_relu_gn_dilation(pool4, filters=256)
    conv5 = conv3d_relu_gn_dilation(conv5, filters=256)
    dsn5 = dsn_block(conv5, ks_stride=16)
    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv6 = GroupNormalization(groups=32)(deconv6)

    up6 = concatenate([deconv6, conv4], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up6, filters=128)
    conv6 = conv3d_relu_gn_dilation(conv6, filters=128)

    dsn6 = dsn_block(conv6, ks_stride=8)
    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=32)(deconv7)

    up7 = concatenate([deconv7, conv3], axis=-1)
    conv7 = conv3d_relu_gn_dilation(up7, filters=64)
    conv7 = conv3d_relu_gn_dilation(conv7, filters=64)

    dsn7 = dsn_block(conv7, ks_stride=4)
    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=16)(deconv8)

    up8 = concatenate([deconv8, conv2], axis=-1)
    conv8 = conv3d_relu_gn_dilation(up8, filters=32)
    conv8 = conv3d_relu_gn_dilation(conv8, filters=32)

    dsn8 = dsn_block(conv8, ks_stride=2)
    deconv9 = Conv3DTranspose(16, 2, strides=2, padding='valid', activation='relu')(conv8)
    deconv9 = GroupNormalization(groups=16)(deconv9)

    up9 = concatenate([deconv9, conv1], axis=-1)
    conv9 = conv3d_relu_gn_dilation(up9, filters=16)
    conv9 = conv3d_relu_gn_dilation(conv9, filters=16)

    dsn9 = dsn_block(conv9, ks_stride=1)
    act = concatenate([dsn1, dsn2, dsn3, dsn4, dsn5, dsn6, dsn7, dsn8, dsn9], axis=-1)
    act = Conv3D(classes, 1, activation='softmax')(act)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def zelin(shape, classes=2):  # modifying...
    inputs = Input(shape)
    conv1 = conv3d_relu_gn_dilation(inputs, filters=32, kernel_size=3)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    # pool1 = Dropout(rate=0.5)(pool1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_relu_gn_dilation(pool1, filters=64)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    # pool2 = Dropout(rate=0.5)(pool2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_relu_gn_dilation(pool2, filters=128)
    print "conv3 shape:", conv3.shape
    conv3 = res_block(conv3, 5, filters=128)

    deconv1 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv3)
    deconv1 = GroupNormalization(groups=32)(deconv1)

    up1 = concatenate([deconv1, conv2], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up1, filters=64)

    deconv2 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv2 = GroupNormalization(groups=32)(deconv2)

    up2 = concatenate([deconv2, conv1], axis=-1)
    conv4 = conv3d_relu_gn_dilation(up2, filters=32)

    act = Conv3D(classes, 1, activation='softmax')(conv4)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=generalised_dice_loss_3d, metrics=[dice_coef])
    return model


def light_resnet(shape, classes=2):
    """
    :param shape:
    :param classes:
    :return:
    """
    inputs = Input(shape)
    # x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(inputs, filters=32)
    # conv1 = res_block(conv1, number=1, filters=32)
    conv1 = conv3d_relu_gn_dilation(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    # pool1 = conv3d_relu_gn_dilation(conv1, filters=32, kernel_size=2, padding='valid', strides=2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3d_relu_gn_dilation(pool1, filters=64)
    # conv2 = res_block(conv2, number=1, filters=64)
    conv2 = conv3d_relu_gn_dilation(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    # pool2 = conv3d_relu_gn_dilation(conv2, filters=64, kernel_size=2, padding='valid', strides=2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3d_relu_gn_dilation(pool2, filters=128)
    # conv3 = res_block(conv3, number=1, filters=128)
    conv3 = conv3d_relu_gn_dilation(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    # pool3 = conv3d_relu_gn_dilation(conv3, filters=128, kernel_size=2, padding='valid', strides=2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3d_relu_gn_dilation(pool3, filters=256)
    # conv4 = res_block(conv4, number=1, filters=256)
    conv4 = conv3d_relu_gn_dilation(conv4, filters=256)
    print "conv4 shape:", conv4.shape
    deconv4 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv4)
    deconv4 = GroupNormalization()(deconv4)
    up4 = concatenate([deconv4, conv3], axis=-1)
    # up4 = add([deconv4, conv3])
    conv5 = conv3d_relu_gn_dilation(up4, filters=128)
    conv5 = res_block(conv5, 1, filters=128)

    deconv5 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv5 = GroupNormalization()(deconv5)
    up5 = concatenate([deconv5, conv2], axis=-1)
    # up5 = add([deconv5, conv2])
    conv6 = conv3d_relu_gn_dilation(up5, filters=64)
    conv6 = res_block(conv6, 1, filters=64)

    deconv6 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv6 = GroupNormalization()(deconv6)
    up6 = concatenate([deconv6, conv1], axis=-1)
    # up6 = add([deconv6, conv1])
    conv7 = conv3d_relu_gn_dilation(up6, filters=32)
    conv7 = res_block(conv7, 1, filters=32)

    act = Conv3D(classes, 1, activation='softmax', padding='same', name='Segment')(conv7)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=dice_coef_loss,
                  metrics={'Segment': dice_coef})
    return model


def best_79(shape, classes=2):
    """
    :param shape:
    :param classes:
    :return:
    """
    inputs = Input(shape)
    # x = CoordinateChannel3D()(inputs)
    conv1 = conv3d_relu_gn_dilation(inputs, filters=32)
    # conv1 = res_block(conv1, number=1, filters=32)
    conv1 = conv3d_relu_gn_dilation(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    # pool1 = ZeroPadding3D(padding=1)(conv1)
    # pool1 = conv3d_relu_gn_dilation(pool1, filters=32, kernel_size=4, padding='valid', strides=2)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3d_relu_gn_dilation(pool1, filters=64)
    # conv2 = res_block(conv2, number=1, filters=64)
    conv2 = conv3d_relu_gn_dilation(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    # pool2 = ZeroPadding3D(padding=1)(conv2)
    # pool2 = conv3d_relu_gn_dilation(pool2, filters=64, kernel_size=3, padding='valid', strides=2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3d_relu_gn_dilation(pool2, filters=128)
    # conv3 = res_block(conv3, number=1, filters=128)
    conv3 = conv3d_relu_gn_dilation(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    # pool3 = ZeroPadding3D(padding=1)(conv3)
    # pool3 = conv3d_relu_gn_dilation(pool3, filters=128, kernel_size=3, padding='valid', strides=2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3d_relu_gn_dilation(pool3, filters=256)
    # conv4 = res_block(conv4, number=2, filters=256)
    conv4 = conv3d_relu_gn_dilation(conv4, filters=256)
    print "conv4 shape:", conv4.shape
    # act2 = GlobalAveragePooling3D()(conv4)
    # print 'act2 shape: ', act2.shape
    # act2 = Dense(256, activation='relu')(act2)
    # act2 = Dense(128, activation='relu')(act2)
    # act2 = Dense(classes, activation='softmax', name='Classify')(act2)
    deconv4 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv4)
    deconv4 = GroupNormalization(groups=32)(deconv4)
    up4 = concatenate([deconv4, conv3], axis=-1)
    conv5 = conv3d_relu_gn_dilation(up4, filters=128)
    # conv5 = res_block(conv5, 1, filters=128)
    conv5 = conv3d_relu_gn_dilation(conv5, filters=128)

    deconv5 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv5 = GroupNormalization(groups=32)(deconv5)
    up5 = concatenate([deconv5, conv2], axis=-1)
    conv6 = conv3d_relu_gn_dilation(up5, filters=64)
    # conv6 = res_block(conv6, 1, filters=64)
    conv6 = conv3d_relu_gn_dilation(conv6, filters=64)

    deconv6 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv6 = GroupNormalization(groups=32)(deconv6)
    up6 = concatenate([deconv6, conv1], axis=-1)
    conv7 = conv3d_relu_gn_dilation(up6, filters=32)
    # conv7 = res_block(conv7, 1, filters=32)
    conv7 = conv3d_relu_gn_dilation(conv7, filters=32)

    act = Conv3D(classes, 1, activation='softmax', padding='same', name='Segment')(conv7)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)

    # model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
    #               loss=[dice_coef_loss, 'categorical_crossentropy'],
    #               metrics={'Segment': dice_coef, 'Classify': 'accuracy'},
    #               loss_weights=[1, 0.5])  # 1, 0.5
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=dice_coef_loss,
                  metrics={'Segment': dice_coef})
    return model


def elegant_model(shape, classes=2):
    inputs = Input(shape)
    conv1 = conv3d_relu_gn_dilation(inputs, filters=16)
    conv1 = res_block(conv1, number=1, filters=16)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3d_relu_gn_dilation(pool1, filters=32)
    conv2 = res_block(conv2, number=2, filters=32)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3d_relu_gn_dilation(pool2, filters=64)
    conv3 = res_block(conv3, number=3, filters=64)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3d_relu_gn_dilation(pool3, filters=128)
    conv4 = res_block(conv4, number=4, filters=128)
    print 'conv4 shape:', conv4.shape
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_relu_gn_dilation(pool4, filters=256)
    conv5 = res_block(conv5, number=2, filters=256)
    conv5 = convolutional_block_attention_module(conv5, 256)
    conv5 = res_block(conv5, number=2, filters=256)
    print 'conv5 shape:', conv5.shape

    deconv4 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv4 = GroupNormalization()(deconv4)
    up4 = concatenate([deconv4, conv4], axis=-1)
    up4 = conv3d_relu_gn_dilation(up4, filters=128)
    up4 = res_block(up4, 4, filters=128)

    deconv3 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up4)
    deconv3 = GroupNormalization()(deconv3)
    up3 = concatenate([deconv3, conv3], axis=-1)
    up3 = conv3d_relu_gn_dilation(up3, filters=64)
    up3 = res_block(up3, 3, filters=64)

    deconv2 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up3)
    deconv2 = GroupNormalization()(deconv2)
    up2 = concatenate([deconv2, conv2], axis=-1)
    up2 = conv3d_relu_gn_dilation(up2, filters=32)
    up2 = res_block(up2, 2, filters=32)

    deconv1 = Conv3DTranspose(16, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up2)
    deconv1 = GroupNormalization(groups=16)(deconv1)
    up1 = concatenate([deconv1, conv1], axis=-1)
    up1 = conv3d_relu_gn_dilation(up1, filters=16)
    up1 = res_block(up1, 1, filters=16)
    # conv7 = conv3d_relu_gn_dilation(conv7, filters=16)

    act = Conv3D(classes, 1, activation='softmax', padding='same', name='Segment')(up1)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=dice_coef_loss,
                  metrics={'Segment': dice_coef})
    return model


def best_shallow(shape, classes=2):
    inputs = Input(shape)
    conv1 = conv3d_relu_gn_dilation(inputs, filters=16)
    conv1 = res_block(conv1, number=1, filters=16)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3d_relu_gn_dilation(pool1, filters=32)
    conv2 = res_block(conv2, number=2, filters=32)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3d_relu_gn_dilation(pool2, filters=64)
    conv3 = res_block(conv3, number=3, filters=64)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3d_relu_gn_dilation(pool3, filters=128)
    conv4 = res_block(conv4, number=4, filters=128)
    print 'conv4 shape:', conv4.shape
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_relu_gn_dilation(pool4, filters=256)
    conv5 = res_block(conv5, number=5, filters=256)
    print 'conv5 shape:', conv5.shape

    deconv4 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv5)
    deconv4 = GroupNormalization()(deconv4)
    up4 = concatenate([deconv4, conv4], axis=-1)
    up4 = convolutional_block_attention_module(up4, 256)
    up4 = conv3d_relu_gn_dilation(up4, filters=128)
    up4 = res_block(up4, 4, filters=128)

    deconv3 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up4)
    deconv3 = GroupNormalization()(deconv3)
    up3 = concatenate([deconv3, conv3], axis=-1)
    up3 = convolutional_block_attention_module(up3, 128)
    up3 = conv3d_relu_gn_dilation(up3, filters=64)
    up3 = res_block(up3, 3, filters=64)

    deconv2 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up3)
    deconv2 = GroupNormalization()(deconv2)
    up2 = concatenate([deconv2, conv2], axis=-1)
    up2 = convolutional_block_attention_module(up2, 64)
    up2 = conv3d_relu_gn_dilation(up2, filters=32)
    up2 = res_block(up2, 2, filters=32)

    deconv1 = Conv3DTranspose(16, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(up2)
    deconv1 = GroupNormalization(groups=16)(deconv1)
    up1 = concatenate([deconv1, conv1], axis=-1)
    up1 = convolutional_block_attention_module(up1, 32)
    up1 = conv3d_relu_gn_dilation(up1, filters=16)
    up1 = res_block(up1, 1, filters=16)
    # conv7 = conv3d_relu_gn_dilation(conv7, filters=16)

    act = Conv3D(classes, 1, activation='softmax', padding='same', name='Segment')(up1)
    print 'act shape:', act.shape
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=dice_coef_loss,
                  metrics={'Segment': dice_coef})
    return model


def deeplabv2(shape, classes=2):
    inputs = Input(shape)
    conv1 = conv3d_relu_gn_without_dilation(inputs, 32, 3)
    conv1 = conv3d_relu_gn_without_dilation(conv1, 32, 3)
    pool1 = MaxPooling3D(3, 2, padding='same')(conv1)

    conv2 = conv3d_relu_gn_without_dilation(pool1, 64, 3)
    conv2 = conv3d_relu_gn_without_dilation(conv2, 64, 3)
    pool2 = MaxPooling3D(3, 2, padding='same')(conv2)

    conv3 = conv3d_relu_gn_without_dilation(pool2, 128, 3)
    conv3 = conv3d_relu_gn_without_dilation(conv3, 128, 3)
    pool3 = MaxPooling3D(3, 2, padding='same')(conv3)

    conv4 = conv3d_relu_gn_without_dilation(pool3, 256, 3)
    conv4 = conv3d_relu_gn_without_dilation(conv4, 256, 3)
    pool4 = MaxPooling3D(3, 2, padding='same')(conv4)

    conv5 = conv3d_relu_gn_dilation(pool4, 128, 3, dilation_rate=2)
    conv5 = conv3d_relu_gn_dilation(conv5, 128, 3, dilation_rate=2)
    pool5 = MaxPooling3D(3, 2, padding='same')(conv5)

    aspp1 = conv3d_relu_gn_dilation(pool5, 512, 3, dilation_rate=2)
    aspp1 = conv3d_relu_gn_dilation(aspp1, 512, 1)
    aspp1 = conv3d_relu_gn_dilation(aspp1, classes, 1)

    aspp2 = conv3d_relu_gn_dilation(pool5, 512, 3, dilation_rate=3)
    aspp2 = conv3d_relu_gn_dilation(aspp2, 512, 1)
    aspp2 = conv3d_relu_gn_dilation(aspp2, classes, 1)

    aspp3 = conv3d_relu_gn_dilation(pool5, 512, 3, dilation_rate=4)
    aspp3 = conv3d_relu_gn_dilation(aspp3, 512, 1)
    aspp3 = conv3d_relu_gn_dilation(aspp3, classes, 1)

    aspp4 = conv3d_relu_gn_dilation(pool5, 512, 3, dilation_rate=5)
    aspp4 = conv3d_relu_gn_dilation(aspp4, 512, 1)
    aspp4 = conv3d_relu_gn_dilation(aspp4, classes, 1)

    aspp = add([aspp1, aspp2, aspp3, aspp4])
    act = Softmax()(aspp)
    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model


def Nest_Net(shape, classes=1, deep_supervision=True):
    # nb_filter = [8, 16, 32, 64, 128]
    nb_filter = [16, 64, 128, 256]
    img_input = Input(shape, name='main_input')
    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    print 'conv1_1 shape:', conv1_1.shape
    pool1 = MaxPooling3D(2, strides=2, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling3D(2, strides=2, name='pool2')(conv2_1)

    up1_2 = Conv3DTranspose(nb_filter[0], 2, strides=2, name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12')
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling3D(2, strides=2, name='pool3')(conv3_1)

    up2_2 = Conv3DTranspose(nb_filter[1], 2, strides=2, name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22')
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv3DTranspose(nb_filter[0], 2, strides=2, name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13')
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])

    up3_2 = Conv3DTranspose(nb_filter[2], 2, strides=2, name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32')
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv3DTranspose(nb_filter[1], 2, strides=2, name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23')
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv3DTranspose(nb_filter[0], 2, strides=2, name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_3, conv1_2], name='merge14')
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv3D(classes, 1, activation='sigmoid', name='output_1',
                              kernel_initializer='he_normal', padding='same')(conv1_2)
    nestnet_output_2 = Conv3D(classes, 1, activation='sigmoid', name='output_2',
                              kernel_initializer='he_normal', padding='same')(conv1_3)
    nestnet_output_3 = Conv3D(classes, 1, activation='sigmoid', name='output_3',
                              kernel_initializer='he_normal', padding='same')(conv1_4)
    print 'nestnet_output_1 shape:', nestnet_output_1.shape
    print 'nestnet_output_2 shape:', nestnet_output_2.shape
    print 'nestnet_output_3 shape:', nestnet_output_3.shape
    # nestnet_output = concatenate([nestnet_output_1, nestnet_output_2, nestnet_output_3])
    # nestnet_output = concatenate([conv1_2, conv1_3, conv1_4])
    # nestnet_output = Conv3D(classes, 1, activation='sigmoid', name='output',
    #                         kernel_initializer='he_normal')(nestnet_output)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3])
    else:
        model = Model(input=img_input, output=[nestnet_output_3])

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]),
                  loss=[dice_coef_loss, dice_coef_loss, dice_coef_loss],
                  metrics={'output_1': dice_coef, 'output_2': dice_coef, 'output_3': dice_coef})
    return model


# Module
def conv3d_relu_gn_without_dilation(x, filters=32, kernel_size=3, padding='same', strides=1):
    out = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(x)
    out = GroupNormalization(groups=filters)(out)
    # out = BatchNormalization()(out)
    return out


def conv3d_prelu(x, filters=32, kernel_size=3, padding='same', strides=1):
    out = Conv3D(filters=filters, kernel_size=kernel_size,
                 strides=strides, padding=padding,
                 activation='relu')(x)
    # out = PReLU()(out)
    # out = GroupNormalization(groups=16)(out)
    out = BatchNormalization()(out)
    return out


def conv3d_relu_gn_dilation(x, filters=32, kernel_size=3, dilation_rate=1, padding='same', strides=1):
    out = Conv3D(filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding, activation='relu',
                 strides=strides)(x)
    if filters < 32:
        out = GroupNormalization(groups=filters)(out)
    else:
        out = GroupNormalization()(out)
    # out = BatchNormalization()(out)
    # out = convolutional_block_attention_module(out, filters)
    return out


def squeeze_excitation_layer(x, out_dim):
    squeeze = GlobalAveragePooling3D()(x)
    excitation = Dense(units=out_dim // 16, activation='relu')(squeeze)
    excitation = Dense(units=out_dim, activation='sigmoid')(excitation)
    # excitation = Conv3D(filters=out_dim // 16, kernel_size=1, activation='relu')(squeeze)
    # excitation = Conv3D(filters=out_dim, kernel_size=1, activation='sigmoid')(excitation)
    scale = multiply([x, excitation])

    return scale


def convolutional_block_attention_module(x, out_dim):
    """
    :param x:
    :param out_dim:
    :return:
    :reference: Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module[J]. 2018.
    """
    # Channel module
    avg_squeeze = GlobalAveragePooling3D()(x)
    max_squeeze = GlobalMaxPooling3D()(x)
    # shared MLP
    dense1 = Dense(units=out_dim // 16, activation='relu')
    dense2 = Dense(units=out_dim)
    avg_excitation = dense1(avg_squeeze)
    max_excitation = dense1(max_squeeze)
    avg_excitation = dense2(avg_excitation)
    max_excitation = dense2(max_excitation)
    excitation = add([avg_excitation, max_excitation])
    excitation = Activation('sigmoid')(excitation)
    channel_attention = multiply([x, excitation])

    # Spatial module
    max_spatial = Lambda(K.max,
                         arguments={'axis': -1, 'keepdims': True})(channel_attention)
    avg_spatial = Lambda(K.mean,
                         arguments={'axis': -1, 'keepdims': True})(channel_attention)
    spatial = concatenate([max_spatial, avg_spatial])
    spatial = Conv3D(1, 1, padding='same', activation='sigmoid')(spatial)
    spatial = multiply([channel_attention, spatial])

    return spatial


def res_block(input, number=1, filters=32):
    for _ in xrange(number):
        output = conv3d_relu_gn_dilation(input, filters=filters/2, kernel_size=1)
        output = conv3d_relu_gn_dilation(output, filters=filters/2, kernel_size=3)
        output = conv3d_relu_gn_dilation(output, filters=filters, kernel_size=1)
        output = add([input, output])
        input = output

    return input


def dsn_block(input, ks_stride=1):
    """
    Holistically-Nested Edge Detection
    :param input:
    :param ks_stride: kernel size and stride
    :return:
    """
    conv = Conv3D(1, 1)(input)
    deconv = Conv3DTranspose(1, ks_stride, ks_stride)(conv)

    return deconv


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    dropout_ratio = 1
    groups = nb_filter if nb_filter < 32 else 32
    x = Conv3D(nb_filter, kernel_size, activation='relu', name='conv'+stage+'_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_ratio))(input_tensor)
    x = GroupNormalization(groups)(x)
    # x = Dropout(dropout_ratio, name='dp'+stage+'_1')(x)
    x = Conv3D(nb_filter, kernel_size, activation='relu', name='conv'+stage+'_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_ratio))(x)
    x = GroupNormalization(groups)(x)
    # x = Dropout(dropout_ratio, name='dp'+stage+'_2')(x)

    return x


##
def dice_coef(y_true, y_pred):
    if y_pred.get_shape()[-1] == 2:
        y_pred_f = K.flatten(y_pred[..., -1])
    else:
        y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def jaccard_loss(y_true, y_pred):
    if y_pred.get_shape()[-1] == 2:
        y_pred_f = K.flatten(y_pred[..., -1])
    else:
        y_pred_f = K.flatten(y_pred)
    smooth = 1e-5
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return -intersection / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)  # + 0.5 * smoothness_loss_mine(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    if y_pred.get_shape()[-1] == 2:
        y_pred = y_pred[..., -1][..., None]
    smooth = 1e-5
    alpha = 0.3
    beta = 0.7
    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred
    p1 = ones - p0
    g0 = y_true
    g1 = ones - y_true
    num = tf.reduce_sum(p0 * g0, axis=(0, 1, 2, 3))
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=(0, 1, 2, 3)) + \
          beta * tf.reduce_sum(p1 * g0, axis=(0, 1, 2, 3)) + smooth
    loss = -tf.reduce_sum(num / den)
    return loss


def topology_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred: haven't been activated by softmax
    :return:
    """
    y_true = tf.cast(y_true, tf.uint8)
    one_hot_gt_masks = tf.squeeze(tf.one_hot(y_true, depth=2, axis=4), axis=-1)
    a_0 = tf.slice(y_pred, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1])
    a_1 = tf.slice(y_pred, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1])
    e_0 = tf.exp(a_0)
    e_1 = tf.exp(a_1)
    z = e_0 + e_1 + 1e-10
    p_0 = e_0 / z
    p_1 = e_1 / z
    prob = tf.concat([p_0, p_1], 4)
    # prob_y = prob * one_hot_gt_masks
    # loss_per_pixel = - tf.reduce_sum(tf.log(prob_y + 1e-10), axis=
    loss_per_pixel = - tf.reduce_mean(tf.reduce_sum(one_hot_gt_masks * tf.log(prob + 1e-10), axis=4))
    # loss = tf.reduce_mean(loss_per_pixel)
    return loss_per_pixel


def smoothness_loss(gt_masks, pred_prob):
    gt_mask_5d = tf.cast(gt_masks, tf.float32)
    filter_x = np.array([1, -1], dtype=np.float32).reshape([1, 2, 1, 1, 1])
    filter_y = np.array([1, -1], dtype=np.float32).reshape([2, 1, 1, 1, 1])
    filter_z = np.array([1, -1], dtype=np.float32).reshape([1, 1, 2, 1, 1])
    # boundary ground truth
    b_x = tf.nn.conv3d(gt_mask_5d, filter_x, strides=[1, 1, 1, 1, 1], padding="SAME")
    b_y = tf.nn.conv3d(gt_mask_5d, filter_y, strides=[1, 1, 1, 1, 1], padding="SAME")
    b_z = tf.nn.conv3d(gt_mask_5d, filter_z, strides=[1, 1, 1, 1, 1], padding="SAME")

    one_hot_gt_masks = tf.squeeze(tf.one_hot(tf.cast(gt_masks, tf.uint8), depth=2, axis=4), axis=-1)
    # # compute log probability of ground truth
    prob_gt = tf.reduce_sum(one_hot_gt_masks * tf.log(pred_prob + 1e-5), axis=4, keep_dims=True)

    prob_diff_x = tf.abs(tf.nn.conv3d(prob_gt, filter_x, strides=[1, 1, 1, 1, 1], padding="SAME"))
    prob_diff_y = tf.abs(tf.nn.conv3d(prob_gt, filter_y, strides=[1, 1, 1, 1, 1], padding="SAME"))
    prob_diff_z = tf.abs(tf.nn.conv3d(prob_gt, filter_z, strides=[1, 1, 1, 1, 1], padding="SAME"))

    prob_diff_x1 = tf.where(tf.equal(b_x, 0), prob_diff_x, tf.zeros_like(prob_diff_x))
    prob_diff_y1 = tf.where(tf.equal(b_y, 0), prob_diff_y, tf.zeros_like(prob_diff_y))
    prob_diff_z1 = tf.where(tf.equal(b_z, 0), prob_diff_z, tf.zeros_like(prob_diff_z))

    loss_per_pixel = tf.reduce_mean(tf.squeeze(prob_diff_x1 + prob_diff_y1 + prob_diff_z1, 4))
    # loss = tf.reduce_mean(prob_diff_x) + tf.reduce_mean(prob_diff_y)
    return loss_per_pixel


def smoothness_loss_mine(gt_masks, pred_prob):
    filter_x = np.array([1, -1], dtype=np.float32).reshape([1, 2, 1, 1, 1])
    filter_y = np.array([1, -1], dtype=np.float32).reshape([2, 1, 1, 1, 1])
    filter_z = np.array([1, -1], dtype=np.float32).reshape([1, 1, 2, 1, 1])

    one_hot_gt_masks = tf.squeeze(tf.one_hot(tf.cast(gt_masks, tf.uint8), depth=2, axis=4), axis=-1)
    # # compute log probability of ground truth
    prob_gt = tf.reduce_sum(one_hot_gt_masks * tf.log(pred_prob + 1e-5), axis=4, keep_dims=True)

    prob_diff_x = tf.abs(tf.nn.conv3d(prob_gt, filter_x, strides=[1, 1, 1, 1, 1], padding="SAME"))
    prob_diff_y = tf.abs(tf.nn.conv3d(prob_gt, filter_y, strides=[1, 1, 1, 1, 1], padding="SAME"))
    prob_diff_z = tf.abs(tf.nn.conv3d(prob_gt, filter_z, strides=[1, 1, 1, 1, 1], padding="SAME"))

    prob_diff_x1 = tf.where(tf.equal(gt_masks, 1), prob_diff_x, tf.zeros_like(prob_diff_x))
    prob_diff_y1 = tf.where(tf.equal(gt_masks, 1), prob_diff_y, tf.zeros_like(prob_diff_y))
    prob_diff_z1 = tf.where(tf.equal(gt_masks, 1), prob_diff_z, tf.zeros_like(prob_diff_z))

    loss_per_pixel = tf.reduce_mean(tf.squeeze(prob_diff_x1 + prob_diff_y1 + prob_diff_z1, 4))
    # loss = tf.reduce_mean(prob_diff_x) + tf.reduce_mean(prob_diff_y)
    return loss_per_pixel


def generalised_dice_loss_3d(y_true, y_pred):
    smooth = 1e-5
    w = tf.reduce_sum(y_true, axis=(0, 1, 2, 3))
    w = 1 / (w ** 2 + smooth)

    numerator = y_true * y_pred[..., -1][..., None]
    numerator = w * tf.reduce_sum(numerator, axis=(0, 1, 2, 3, 4))
    numerator = tf.reduce_sum(numerator)

    denominator = y_pred[..., -1][..., None] + y_true
    denominator = w * tf.reduce_sum(denominator, axis=(0, 1, 2, 3, 4))
    denominator = tf.reduce_sum(denominator)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = 1 - gen_dice_coef
    return loss