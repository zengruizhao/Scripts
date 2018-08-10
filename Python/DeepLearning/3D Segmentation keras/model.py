from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Conv3DTranspose,GlobalAveragePooling3D, multiply, add
from keras.layers.core import Activation
from keras.optimizers import Adam,Adadelta, SGD
from config import config
from keras import backend as K
from keras import regularizers
from keras.layers import concatenate, Conv2DTranspose, add
from coord import CoordinateChannel3D
from group_norm import GroupNormalization
from SwitchableNormalization import SwitchableNormalization
import numpy as np
from keras.layers.merge import concatenate


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
    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4],axis=-1)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    # up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3],axis=-1)
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


def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + config["smooth"]) / (K.sum(y_true_f) + K.sum(y_pred_f) + config["smooth"])


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv(shape, classes=2):
    inputs = Input(shape)
    x = CoordinateChannel3D()(inputs)
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:",pool1.shape

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
    conv1 = conv3d_gn_relu(x, filters=32)
    print "conv1 shape:", conv1.shape
    conv1 = conv3d_gn_relu(conv1, filters=32)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:", pool1.shape

    conv2 = conv3d_gn_relu(pool1, filters=64)
    print "conv2 shape:", conv2.shape
    conv2 = conv3d_gn_relu(conv2, filters=64)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = conv3d_gn_relu(pool2, filters=128)
    print "conv3 shape:", conv3.shape
    conv3 = conv3d_gn_relu(conv3, filters=128)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = conv3d_gn_relu(pool3, filters=256)
    print "conv4 shape:", conv4.shape
    conv4 = conv3d_gn_relu(conv4, filters=256)
    print "conv4 shape:", conv4.shape

    deconv6 = Conv3DTranspose(128, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv4)
    deconv6 = GroupNormalization(groups=32)(deconv6)
    
    up6 = concatenate([deconv6, conv3], axis=-1)
    conv6 = conv3d_gn_relu(up6, filters=128)
    conv6 = conv3d_gn_relu(conv6, filters=128)

    deconv7 = Conv3DTranspose(64, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv6)
    deconv7 = GroupNormalization(groups=32)(deconv7)

    up7 = concatenate([deconv7, conv2], axis=-1)
    conv7 = conv3d_gn_relu(up7, filters=64)
    conv7 = conv3d_gn_relu(conv7, filters=64)

    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',
                              activation='relu')(conv7)
    deconv8 = GroupNormalization(groups=32)(deconv8)

    up8 = concatenate([deconv8, conv1], axis=-1)
    conv8 = conv3d_gn_relu(up8, filters=32)
    conv8 = conv3d_gn_relu(conv8, filters=32)

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

    conv5 = Conv3D(classes, (1, 1, 1), activation='relu')(deconv3)

    conv5 = BatchNormalization()(conv5)
    act = Conv3D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_GN_best(shape, classes=2): # Need more memory
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
    deconv7 = GroupNormalization(groups=32)(deconv7)
    deconv7 = Conv3D(64, 3, activation='relu', padding='same')(deconv7)
    deconv7 = GroupNormalization(groups=32)(deconv7)
    conv_conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv_conv3 = GroupNormalization(groups=32)(conv_conv3)
    up7 = concatenate([deconv7, conv_conv3],axis=-1)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = GroupNormalization(groups=32)(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = GroupNormalization(groups=32)(conv7)


    deconv8 = Conv3DTranspose(32, 2,
                              strides=(2, 2, 2),
                              padding='valid',)(conv7)
    deconv8 = GroupNormalization(groups=32)(deconv8)
    deconv8 = Conv3D(32, 3, activation='relu', padding='same')(deconv8)
    deconv8 = GroupNormalization(groups=32)(deconv8)
    conv_conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv_conv2 = GroupNormalization(groups=32)(conv_conv2)
    up8 = concatenate([deconv8, conv_conv2], axis=-1)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = GroupNormalization(groups=32)(conv8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = GroupNormalization(groups=32)(conv8)


    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = GroupNormalization(groups=2)(conv10)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=act)

    model.compile(optimizer=Adam(lr=config["initial_learning_rate"]), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def deconv_conv_unet_model_3d_conv_add_48_dalitaion_coordconv_SN(shape, classes=2):# reduced feature map
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


def conv3d_gn_relu(x, filters=32):
    out = Conv3D(filters, (5, 3, 3), padding='same', activation='relu')(x)
    out = GroupNormalization(groups=32)(out)
    return out

