#coding:utf-8
import tensorflow as tf
import tensorlayer as tl
from keras import backend as K
K.set_learning_phase(1)
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Add,Dropout
from keras.layers.normalization import  BatchNormalization

from keras.optimizers import Adam,Adadelta, RMSprop
from config import config
from keras.layers import concatenate, Conv2DTranspose, add
from keras_contrib.layers import Deconvolution3D
from keras.regularizers import l2
import numpy as np
from keras.layers import *

def u_net_3d_mw(x,classes,shape):
    input = tl.layers.InputLayer(x,name='input')
    with tf.name_scope('conv1'):
        conv1_1 = tl.layers.Conv3dLayer(input,act=tf.nn.relu,shape=[3,3,3,1,32],strides=[1,1,1,1,1],padding='SAME',name='conv1_1')
        conv1_2 = tl.layers.Conv3dLayer(conv1_1,act=tf.nn.relu,shape=[3,3,3,32,32],strides=[1,1,1,1,1],padding='SAME',name='conv1_2')
        pool1 = tl.layers.MaxPool3d(conv1_2,filter_size=[2,2,2],strides=[2,2,2],name='pool1')
    with tf.name_scope('conv2'):
        conv2_1 = tl.layers.Conv3dLayer(pool1,act=tf.nn.relu,shape=[3,3,3,32,64],strides=[1,1,1,1,1],padding='SAME',name='conv2_1')
        conv2_2 = tl.layers.Conv3dLayer(conv2_1, act=tf.nn.relu, shape=[3, 3, 3, 64, 64], strides=[1,1,1,1,1],padding='SAME', name='conv2_2')
        pool2 = tl.layers.MaxPool3d(conv2_2,filter_size=[2,2,2],strides=[2,2,2],name='pool2')
    with tf.name_scope('conv3'):
        conv3_1 = tl.layers.Conv3dLayer(pool2, act=tf.nn.relu, shape=[3, 3, 3, 64,128], strides=[1,1,1,1,1],padding='SAME', name='conv3_1')
        conv3_2 = tl.layers.Conv3dLayer(conv3_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1,1,1,1,1],padding='SAME', name='conv3_2')
        pool3 = tl.layers.MaxPool3d(conv3_2, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool3')
    with tf.name_scope('conv4'):
        conv4_1 = tl.layers.Conv3dLayer(pool3, act=tf.nn.relu, shape=[3, 3, 3, 128,256],strides=[1,1,1,1,1], padding='SAME', name='conv4_1')
        conv4_2 = tl.layers.Conv3dLayer(conv4_1, act=tf.nn.relu, shape=[3, 3, 3, 256, 256], strides=[1,1,1,1,1],padding='SAME', name='conv4_2')
        pool4 = tl.layers.MaxPool3d(conv4_2, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool4')
    with tf.name_scope('conv5'):
        conv5_1 = tl.layers.Conv3dLayer(pool4, act=tf.nn.relu, shape=[3, 3, 3, 256, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_1')
        conv5_2 = tl.layers.Conv3dLayer(conv5_1, act=tf.nn.relu, shape=[3, 3, 3, 512, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_2')
    with tf.name_scope('up6'):
        up6 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(conv5_2,act=tf.nn.relu,shape=[3,3,3,256,512],output_shape=[1,shape[2]/16,shape[1]/16,shape[0]/16,256],name='upsample6'),conv4_2],concat_dim=-1,name='up6')
        upconv6_1 = tl.layers.Conv3dLayer(up6,act=tf.nn.relu,shape=[3,3,3,512,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_1')
        upconv6_2 = tl.layers.Conv3dLayer(upconv6_1,act=tf.nn.relu,shape=[3,3,3,256,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_2')
    with tf.name_scope('up7'):
        up7 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv6_2, act=tf.nn.relu, shape=[3, 3, 3, 128,256],output_shape=[1, shape[2] / 8, shape[1] / 8, shape[0] / 8,128], name='upsample7'), conv3_2], concat_dim=-1,name='up7')
        upconv7_1 = tl.layers.Conv3dLayer(up7, act=tf.nn.relu, shape=[3, 3, 3, 256, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_1')
        upconv7_2 = tl.layers.Conv3dLayer(upconv7_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_2')
    with tf.name_scope('up8'):
        up8 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv7_2, act=tf.nn.relu, shape=[3, 3, 3, 64, 128],output_shape=[1, shape[2] / 4, shape[1] / 4, shape[0] / 4,64], name='upsample8'), conv2_2], concat_dim=-1,name='up8')
        upconv8_1 = tl.layers.Conv3dLayer(up8, act=tf.nn.relu, shape=[3, 3, 3, 128, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_1')
        upconv8_2 = tl.layers.Conv3dLayer(upconv8_1, act=tf.nn.relu, shape=[3, 3, 3, 64, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_2')
    with tf.name_scope('up9'):
        up9 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv8_2, act=tf.nn.relu, shape=[3, 3, 3, 32, 64],output_shape=[1, shape[2] / 2, shape[1] / 2, shape[0] / 2,32], name='upsample9'), conv1_2], concat_dim=-1,name='up9')
        upconv9_1 = tl.layers.Conv3dLayer(up9, act=tf.nn.relu, shape=[3, 3, 3, 64, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_1')
        upconv9_2 = tl.layers.Conv3dLayer(upconv9_1, act=tf.nn.relu, shape=[3, 3, 3, 32, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_2')
    with tf.name_scope('up10'):
        conv10 = tl.layers.Conv3dLayer(upconv9_2,act=tf.nn.relu,shape=[3,3,3,32,classes],strides=[1,1,1,1,1],padding='SAME',name='conv10')
    return conv10

def u_net_3d_mw_3(x,classes,shape):
    input = tl.layers.InputLayer(x,name='input')
    with tf.name_scope('conv1'):
        conv1_1 = tl.layers.Conv3dLayer(input,act=tf.nn.relu,shape=[3,3,3,1,64],strides=[1,1,1,1,1],padding='SAME',name='conv1_1')
        conv1_2 = tl.layers.Conv3dLayer(conv1_1,act=tf.nn.relu,shape=[3,3,3,64,64],strides=[1,1,1,1,1],padding='SAME',name='conv1_2')
        pool1 = tl.layers.MaxPool3d(conv1_2,filter_size=[2,2,2],strides=[2,2,2],name='pool1')
    with tf.name_scope('conv2'):
        conv2_1 = tl.layers.Conv3dLayer(pool1,act=tf.nn.relu,shape=[3,3,3,64,128],strides=[1,1,1,1,1],padding='SAME',name='conv2_1')
        conv2_2 = tl.layers.Conv3dLayer(conv2_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1,1,1,1,1],padding='SAME', name='conv2_2')
        pool2 = tl.layers.MaxPool3d(conv2_2,filter_size=[2,2,2],strides=[2,2,2],name='pool2')
    with tf.name_scope('conv5'):
        conv3_1 = tl.layers.Conv3dLayer(pool2, act=tf.nn.relu, shape=[3, 3, 3, 128, 256], strides=[1,1,1,1,1],padding='SAME', name='conv5_1')
        conv3_2 = tl.layers.Conv3dLayer(conv3_1, act=tf.nn.relu, shape=[3, 3, 3, 256, 256], strides=[1,1,1,1,1],padding='SAME', name='conv5_2')
    with tf.name_scope('up6'):
        up6 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(conv3_2,act=tf.nn.relu,shape=[3,3,3,256,256],output_shape=[1,shape[2]/16,shape[1]/4,shape[0]/4,256],name='upsample6'),conv2_2],concat_dim=-1,name='up6')
        upconv6_1 = tl.layers.Conv3dLayer(up6,act=tf.nn.relu,shape=[3,3,3,256,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_1')
        upconv6_2 = tl.layers.Conv3dLayer(upconv6_1,act=tf.nn.relu,shape=[3,3,3,256,128],strides=[1,1,1,1,1],padding='SAME',name='upconv6_2')
    with tf.name_scope('up9'):
        up9 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv6_2, act=tf.nn.relu, shape=[3, 3, 3, 128, 64],output_shape=[1, shape[2] / 2, shape[1] / 2, shape[0] / 2,32], name='upsample9'), conv1_2], concat_dim=-1,name='up9')
        upconv9_1 = tl.layers.Conv3dLayer(up9, act=tf.nn.relu, shape=[3, 3, 3, 64, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_1')
        upconv9_2 = tl.layers.Conv3dLayer(upconv9_1, act=tf.nn.relu, shape=[3, 3, 3, 32, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_2')
    with tf.name_scope('up10'):
        conv10 = tl.layers.Conv3dLayer(upconv9_2,act=tf.nn.relu,shape=[3,3,3,32,classes],strides=[1,1,1,1,1],padding='SAME',name='conv10')
    return conv10

# def u_net_3d_mw(x,classes,shape):
#     input = tl.layers.InputLayer(x,name='input')
#     with tf.name_scope('conv1'):
#         conv1_1 = tl.layers.Conv3dLayer(input,act=tf.nn.relu,shape=[3,3,3,1,32],strides=[1,1,1,1,1],padding='SAME',name='conv1_1')
#         conv1_2 = tl.layers.Conv3dLayer(conv1_1,act=tf.nn.relu,shape=[3,3,3,32,32],strides=[1,1,1,1,1],padding='SAME',name='conv1_2')
#         pool1 = tl.layers.MaxPool3d(conv1_2,filter_size=[2,2,2],strides=[2,2,2],name='pool1')
#     with tf.name_scope('conv2'):
#         conv2_1 = tl.layers.Conv3dLayer(pool1,act=tf.nn.relu,shape=[3,3,3,32,64],strides=[1,1,1,1,1],padding='SAME',name='conv2_1')
#         conv2_2 = tl.layers.Conv3dLayer(conv2_1, act=tf.nn.relu, shape=[3, 3, 3, 64, 64], strides=[1,1,1,1,1],padding='SAME', name='conv2_2')
#         pool2 = tl.layers.MaxPool3d(conv2_2,filter_size=[2,2,2],strides=[2,2,2],name='pool2')
#     with tf.name_scope('conv3'):
#         conv3_1 = tl.layers.Conv3dLayer(pool2, act=tf.nn.relu, shape=[3, 3, 3, 64,128], strides=[1,1,1,1,1],padding='SAME', name='conv3_1')
#         conv3_2 = tl.layers.Conv3dLayer(conv3_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1,1,1,1,1],padding='SAME', name='conv3_2')
#         pool3 = tl.layers.MaxPool3d(conv3_2, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool3')
#     with tf.name_scope('conv4'):
#         conv4_1 = tl.layers.Conv3dLayer(pool3, act=tf.nn.relu, shape=[3, 3, 3, 128,256],strides=[1,1,1,1,1], padding='SAME', name='conv4_1')
#         conv4_2 = tl.layers.Conv3dLayer(conv4_1, act=tf.nn.relu, shape=[3, 3, 3, 256, 256], strides=[1,1,1,1,1],padding='SAME', name='conv4_2')
#         pool4 = tl.layers.MaxPool3d(conv4_2, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool4')
#     with tf.name_scope('conv5'):
#         conv5_1 = tl.layers.Conv3dLayer(pool4, act=tf.nn.relu, shape=[3, 3, 3, 256, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_1')
#         conv5_2 = tl.layers.Conv3dLayer(conv5_1, act=tf.nn.relu, shape=[3, 3, 3, 512, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_2')
#     with tf.name_scope('up6'):
#         up6 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(conv5_2,act=tf.nn.relu,shape=[3,3,3,256,512],output_shape=[1,shape[2]/16,shape[1]/16,shape[0]/16,256],name='upsample6'),conv4_2],concat_dim=-1,name='up6')
#         upconv6_1 = tl.layers.Conv3dLayer(up6,act=tf.nn.relu,shape=[3,3,3,512,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_1')
#         upconv6_2 = tl.layers.Conv3dLayer(upconv6_1,act=tf.nn.relu,shape=[3,3,3,256,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_2')
#     with tf.name_scope('up7'):
#         up7 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv6_2, act=tf.nn.relu, shape=[3, 3, 3, 128,256],output_shape=[1, shape[2] / 8, shape[1] / 8, shape[0] / 8,128], name='upsample7'), conv3_2], concat_dim=-1,name='up7')
#         upconv7_1 = tl.layers.Conv3dLayer(up7, act=tf.nn.relu, shape=[3, 3, 3, 256, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_1')
#         upconv7_2 = tl.layers.Conv3dLayer(upconv7_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_2')
#     with tf.name_scope('up8'):
#         up8 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv7_2, act=tf.nn.relu, shape=[3, 3, 3, 64, 128],output_shape=[1, shape[2] / 4, shape[1] / 4, shape[0] / 4,64], name='upsample8'), conv2_2], concat_dim=-1,name='up8')
#         upconv8_1 = tl.layers.Conv3dLayer(up8, act=tf.nn.relu, shape=[3, 3, 3, 128, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_1')
#         upconv8_2 = tl.layers.Conv3dLayer(upconv8_1, act=tf.nn.relu, shape=[3, 3, 3, 64, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_2')
#     with tf.name_scope('up9'):
#         up9 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv8_2, act=tf.nn.relu, shape=[3, 3, 3, 32, 64],output_shape=[1, shape[2] / 2, shape[1] / 2, shape[0] / 2,32], name='upsample9'), conv1_2], concat_dim=-1,name='up9')
#         upconv9_1 = tl.layers.Conv3dLayer(up9, act=tf.nn.relu, shape=[3, 3, 3, 64, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_1')
#         upconv9_2 = tl.layers.Conv3dLayer(upconv9_1, act=tf.nn.relu, shape=[3, 3, 3, 32, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_2')
#     with tf.name_scope('up10'):
#         conv10 = tl.layers.Conv3dLayer(upconv9_2,act=tf.nn.relu,shape=[3,3,3,32,classes],strides=[1,1,1,1,1],padding='SAME',name='conv10')
#     return conv10
#
# def resblock(layer,inn,out,name):
#     conv1 = tl.layers.Conv3dLayer(layer, act=tf.nn.relu, shape=[3, 3, 3, inn, out], strides=[1, 1, 1, 1, 1],
#                                   padding='SAME', name=name + '_1')
#     conv2 = tl.layers.Conv3dLayer(conv1, act=tf.nn.relu, shape=[3, 3, 3, out, out], strides=[1, 1, 1, 1, 1],
#                                   padding='SAME', name=name + '_2')
#     concat = tl.layers.ConcatLayer([layer,conv2],concat_dim=-1,name=name+'_caoncat')
#
#     return concat
# def u_net_3d_resnet(x,classes,shape):
#     input = tl.layers.InputLayer(x,name='input')
#     with tf.name_scope('conv1'):
#         conv1_1 = tl.layers.Conv3dLayer(input,act=tf.nn.relu,shape=[3,3,3,1,64],strides=[1,1,1,1,1],padding='SAME',name='conv1_1')
#         res1 = resblock(conv1_1,64,64,'res1')
#         pool1 = tl.layers.MaxPool3d(res1,filter_size=[2,2,2],strides=[2,2,2],name='pool1')
#     with tf.name_scope('conv2'):
#         conv2_1 = tl.layers.Conv3dLayer(pool1,act=tf.nn.relu,shape=[3,3,3,32,64],strides=[1,1,1,1,1],padding='SAME',name='conv2_1')
#         res2 = resblock(conv2_1, 64, 64, 'res2')
#         pool2 = tl.layers.MaxPool3d(res2,filter_size=[2,2,2],strides=[2,2,2],name='pool2')
#     with tf.name_scope('conv3'):
#         conv3_1 = tl.layers.Conv3dLayer(pool2, act=tf.nn.relu, shape=[3, 3, 3, 64,128], strides=[1,1,1,1,1],padding='SAME', name='conv3_1')
#         res3 = resblock(conv3_1, 64, 128, 'res3')
#         pool3 = tl.layers.MaxPool3d(res3, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool3')
#     with tf.name_scope('conv4'):
#         conv4_1 = tl.layers.Conv3dLayer(pool3, act=tf.nn.relu, shape=[3, 3, 3, 128,256],strides=[1,1,1,1,1], padding='SAME', name='conv4_1')
#         res4 = resblock(conv4_1, 128, 256, 'res4')
#         pool4 = tl.layers.MaxPool3d(res4, filter_size=[2, 2, 2], strides=[2, 2, 2], name='pool4')
#     with tf.name_scope('conv5'):
#         conv5_1 = tl.layers.Conv3dLayer(pool4, act=tf.nn.relu, shape=[3, 3, 3, 256, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_1')
#         conv5_2 = tl.layers.Conv3dLayer(conv5_1, act=tf.nn.relu, shape=[3, 3, 3, 512, 512], strides=[1,1,1,1,1],padding='SAME', name='conv5_2')
#     with tf.name_scope('up6'):
#         up6 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(conv5_2,act=tf.nn.relu,shape=[3,3,3,256,512],output_shape=[1,shape[2]/16,shape[1]/16,shape[0]/16,256],name='upsample6'),conv4_2],concat_dim=-1,name='up6')
#         upconv6_1 = tl.layers.Conv3dLayer(up6,act=tf.nn.relu,shape=[3,3,3,512,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_1')
#         upconv6_2 = tl.layers.Conv3dLayer(upconv6_1,act=tf.nn.relu,shape=[3,3,3,256,256],strides=[1,1,1,1,1],padding='SAME',name='upconv6_2')
#     with tf.name_scope('up7'):
#         up7 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv6_2, act=tf.nn.relu, shape=[3, 3, 3, 128,256],output_shape=[1, shape[2] / 8, shape[1] / 8, shape[0] / 8,128], name='upsample7'), conv3_2], concat_dim=-1,name='up7')
#         upconv7_1 = tl.layers.Conv3dLayer(up7, act=tf.nn.relu, shape=[3, 3, 3, 256, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_1')
#         upconv7_2 = tl.layers.Conv3dLayer(upconv7_1, act=tf.nn.relu, shape=[3, 3, 3, 128, 128], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv7_2')
#     with tf.name_scope('up8'):
#         up8 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv7_2, act=tf.nn.relu, shape=[3, 3, 3, 64, 128],output_shape=[1, shape[2] / 4, shape[1] / 4, shape[0] / 4,64], name='upsample8'), conv2_2], concat_dim=-1,name='up8')
#         upconv8_1 = tl.layers.Conv3dLayer(up8, act=tf.nn.relu, shape=[3, 3, 3, 128, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_1')
#         upconv8_2 = tl.layers.Conv3dLayer(upconv8_1, act=tf.nn.relu, shape=[3, 3, 3, 64, 64], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv8_2')
#     with tf.name_scope('up9'):
#         up9 = tl.layers.ConcatLayer([tl.layers.DeConv3dLayer(upconv8_2, act=tf.nn.relu, shape=[3, 3, 3, 32, 64],output_shape=[1, shape[2] / 2, shape[1] / 2, shape[0] / 2,32], name='upsample9'), conv1_2], concat_dim=-1,name='up9')
#         upconv9_1 = tl.layers.Conv3dLayer(up9, act=tf.nn.relu, shape=[3, 3, 3, 64, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_1')
#         upconv9_2 = tl.layers.Conv3dLayer(upconv9_1, act=tf.nn.relu, shape=[3, 3, 3, 32, 32], strides=[1, 1, 1, 1, 1],padding='SAME', name='upconv9_2')
#     with tf.name_scope('up10'):
#         conv10 = tl.layers.Conv3dLayer(upconv9_2,act=tf.nn.relu,shape=[3,3,3,32,classes],strides=[1,1,1,1,1],padding='SAME',name='conv10')
#     return conv10

def resnet_block1(input,scale = 0.17,activatioon_fn = tf.nn.relu,filters=32):#主分支1-3-1,副分支1,两个res模块主要是filter数目不一样

    with tf.variable_scope('branch_0'):
        tower_conv = tf.layers.conv3d(input,filters=filters,kernel_size=(3,3,3),strides=(2,2,2),padding='SAME')

    with tf.variable_scope('branch_1'):
        tower_conv_1_0 = tf.layers.conv3d(input,filters=filters,kernel_size=(1,1,1),padding='SAME',activation=tf.nn.relu)
        tower_conv_1_1 = tf.layers.conv3d(tower_conv_1_0,filters = filters,kernel_size=(3,3,3),padding='SAME')
        tower_conv_1_2 = tf.layers.conv3d(tower_conv_1_1,filters=filters,kernel_size=(3,3,3),padding='SAME')

    mixed = tf.concat([tower_conv,tower_conv_1_2],axis=-1)

    up = tf.layers.conv3d(mixed,filters=filters,kernel_size=(1,1,1))

    input += up*scale
    if activatioon_fn:
        output = activatioon_fn(input)
    return output

def resnet_block2(input,filters=128):#主分支1-3-1,副分支1
    res_conv1 = tf.layers.conv3d(input,filters=filters,kernel_size=(1,1,1),padding='SAME')
    res_conv2 = tf.layers.conv3d(res_conv1, filters=filters*2, kernel_size=(3, 3, 3),strides=(2,2,2), padding='SAME')
    res_conv3 = tf.layers.conv3d(res_conv2, filters=filters*2, kernel_size=(1, 1, 1), padding='SAME')

    branchconv = tf.layers.conv3d(input,filters=filters*2,kernel_size=(1,1,1),strides=(1,1,1),padding='SAME')
    branchconv_pool = tf.layers.max_pooling3d(branchconv,pool_size=(2,2,2),strides=(2,2,2))
    res_add = tf.add(res_conv3, branchconv_pool)
    return res_add

def deeplab_3d(x,classes=2):
    with tf.name_scope('extrect_features_layer'):
        extractconv_1 = tf.layers.conv3d(x,filters=32,kernel_size=(3,3,3),padding='SAME',activation=tf.nn.relu)
        extractconv_2 = tf.layers.conv3d(extractconv_1,filters=128,kernel_size=(3,3,3),padding='SAME',activation=tf.nn.relu)
        extractpool_1 = tf.layers.max_pooling3d(extractconv_2,pool_size=(2,2,2),strides=(2,2,2))
        extractconv_3 = tf.layers.conv3d(extractpool_1, filters=256, kernel_size=(3, 3, 3), padding='SAME',activation=tf.nn.relu)
        extractpool_2 = tf.layers.max_pooling3d(extractconv_3, pool_size=(2, 2, 2), strides=(2, 2, 2))
        res2 = extractpool_2
    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv3d(res2,filters=256,kernel_size=(1,1,1),padding='SAME',activation=tf.nn.relu)
    with tf.name_scope('astroconv1'):
        astroconv1 = tf.layers.conv3d(res2,filters=256,kernel_size=(3,3,3),dilation_rate=(2,2,2),padding='SAME',activation=tf.nn.relu)
    with tf.name_scope('astroconv2'):
        astroconv2 = tf.layers.conv3d(res2,filters=256,kernel_size=(3,3,3),dilation_rate=(4,4,4),padding='SAME',activation=tf.nn.relu)
    with tf.name_scope('astroconv3'):
        astroconv3 = tf.layers.conv3d(res2,filters=256,kernel_size=(3,3,3),dilation_rate=(8,8,8),padding='SAME',activation=tf.nn.relu)
    with tf.name_scope('image_pooling'):
        pool4 = tf.layers.average_pooling3d(res2,pool_size=(1,1,1),strides=(1,1,1),padding='SAME')
    with tf.name_scope('concat_conv'):
        concat = tf.concat([conv1,astroconv1,astroconv2,astroconv3,pool4],axis = -1)
        concat_conv = tf.layers.conv3d(concat,filters=256,kernel_size=(1,1,1),padding='SAME',activation=tf.nn.relu)
        print('concat_conv',concat_conv)
    with tf.name_scope('upsample'):
        upsample = tf.layers.conv3d_transpose(concat_conv,filters=256,kernel_size=(1,1,1),strides=(2,2,2),padding='VALID',activation=tf.nn.relu)
        # upsample = tf.nn.conv3d_transpose(concat_conv,filter=[40,60,24,512,512],output_shape=(2,2,2),strides=(1,1,1))
        print('upsample',upsample)

    with tf.name_scope('decoder'):
        extractconv = tf.layers.conv3d(x, filters=256, kernel_size=(1, 1, 1),strides=(2,2,2), padding='SAME',activation=tf.nn.relu)

    with tf.name_scope('final'):
        concat_up_init = tf.concat([upsample,extractconv],axis=-1)
        concat_up_init_conv = tf.layers.conv3d(concat_up_init,filters=256,kernel_size=(3,3,3),padding='SAME',activation=tf.nn.relu)
        up_final = tf.layers.conv3d_transpose(concat_up_init_conv,filters=1,kernel_size=(1,1,1),strides=(2,2,2),activation=tf.nn.sigmoid)
        print('up_final',up_final.shape)
    return up_final


def batchnormalization_relu(layer):
    batch_norm = tf.layers.batch_normalization(layer)
    activation = tf.nn.relu(batch_norm)
    return activation

def deeplab_3d_1(x,classes=2):
    with tf.name_scope('extrect_features_layer'):
        # extractconv_1 = tf.layers.conv3d(x,filters=32,kernel_size=(3,3,3),padding='SAME')
        # extractconv_1 = batchnormalization_relu(extractconv_1)
        extractconv_2 = tf.layers.conv3d(x,filters=64,kernel_size=(3,3,3),padding='SAME')
        extractconv_2 = batchnormalization_relu(extractconv_2)
        extractpool_1 = tf.layers.max_pooling3d(extractconv_2,pool_size=(2,2,2),strides=(2,2,2))
        extractconv_3 = tf.layers.conv3d(extractpool_1, filters=128, kernel_size=(3, 3, 3), padding='SAME')
        extractconv_3 = batchnormalization_relu(extractconv_3)
        extractpool_2 = tf.layers.max_pooling3d(extractconv_3, pool_size=(2, 2, 2), strides=(2, 2, 2))
        res2 = extractpool_2
    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv3d(res2,filters=128,kernel_size=(1,1,1),padding='SAME')
        conv1 = batchnormalization_relu(conv1)
    with tf.name_scope('astroconv0'):
        astroconv0 = tf.layers.conv3d(res2, filters=128, kernel_size=(3, 3, 3), dilation_rate=(1, 1, 1), padding='SAME')
        astroconv0 = batchnormalization_relu(astroconv0)
    with tf.name_scope('astroconv1'):
        astroconv1 = tf.layers.conv3d(res2,filters=128,kernel_size=(3,5,3),dilation_rate=(2,2,2),padding='SAME')
        astroconv1 = batchnormalization_relu(astroconv1)
    with tf.name_scope('astroconv2'):
        astroconv2 = tf.layers.conv3d(res2,filters=128,kernel_size=(3,3,3),dilation_rate=(5,5,5),padding='SAME')
        astroconv2 = batchnormalization_relu(astroconv2)
    with tf.name_scope('astroconv3'):
        astroconv3 = tf.layers.conv3d(res2,filters=128,kernel_size=(3,3,3),dilation_rate=(8,8,8),padding='SAME')
        astroconv3 = batchnormalization_relu(astroconv3)
    with tf.name_scope('image_pooling'):
        pool4 = tf.layers.average_pooling3d(res2,pool_size=(1,1,1),strides=(1,1,1),padding='SAME')
    with tf.name_scope('concat_conv'):
        concat = tf.concat([conv1,astroconv0,astroconv1,astroconv2,astroconv3,pool4],axis = -1)
        concat_conv = tf.layers.conv3d(concat,filters=128,kernel_size=(3,3,3),padding='SAME')
        concat_conv = batchnormalization_relu(concat_conv)
    with tf.name_scope('upsample'):
        upsample = tf.layers.conv3d_transpose(concat_conv,filters=128,kernel_size=(1,1,1),strides=(2,2,2))
        print ('upsample1',upsample.shape)
        upsample = batchnormalization_relu(upsample)

    with tf.name_scope('upsample_conv_0'):
        upsample_conv_0 = tf.layers.conv3d(upsample,filters=128,kernel_size=(3,3,3),padding='SAME')
        upsample_conv_0 = batchnormalization_relu(upsample_conv_0)

    # with tf.name_scope('upsample_conv_1'):
    #     upsample_conv_1 = tf.layers.conv3d(upsample_conv_0,filters=128,kernel_size=(3,3,3),padding='SAME')
    #     upsample_conv_1 = batchnormalization_relu(upsample_conv_1)
    #     print ('upsample1', upsample_conv_1.shape)

    with tf.name_scope('decoder'):
        extractconv = tf.layers.conv3d(x, filters=128, kernel_size=(1, 1, 1),strides=(2,2,2), padding='SAME')
        extractconv = batchnormalization_relu(extractconv)

    with tf.name_scope('final'):
        concat_up_init = tf.concat([upsample_conv_0,extractconv],axis=-1)
        concat_up_init_conv = tf.layers.conv3d(concat_up_init,filters=64,kernel_size=(3,3,3),padding='SAME')
        concat_up_init_conv = batchnormalization_relu(concat_up_init_conv)
        upsample2 = tf.layers.conv3d_transpose(concat_up_init_conv,filters=64,kernel_size=(1,1,1),strides=(2,2,2))
        upsample2 = batchnormalization_relu(upsample2)
        upsample2_conv_0 = tf.layers.conv3d(upsample2, filters=64, kernel_size=(1, 1, 1), padding='SAME')
        upsample2_conv_0 = batchnormalization_relu(upsample2_conv_0)
        upsample2_conv_1 = tf.layers.conv3d(upsample2_conv_0, filters=1, kernel_size=(1, 1, 1), padding='SAME',activation=tf.nn.sigmoid)

    return upsample2_conv_1





def keras_batchnormalization_relu(layer):
    BN = BatchNormalization()(layer)
    return BN


def deconv_conv_unet_model_3d_conv_add_48_dalitaion(inputs, classes=2):
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(inputs)
    conv1 = keras_batchnormalization_relu(conv1)
    print "conv1 shape:", conv1.shape
    conv1 = Conv3D(32, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv1)
    conv1 = keras_batchnormalization_relu(conv1)
    print "conv1 shape:", conv1.shape
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    print "pool1 shape:",pool1.shape

    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool1)
    conv2 = keras_batchnormalization_relu(conv2)
    print "conv2 shape:", conv2.shape
    conv2 = Conv3D(64, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv2)
    conv2 = keras_batchnormalization_relu(conv2)
    print "conv2 shape:", conv2.shape
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    print "pool2 shape:", pool2.shape

    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool2)
    conv3 = keras_batchnormalization_relu(conv3)
    print "conv3 shape:", conv3.shape
    conv3 = Conv3D(128, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv3)
    conv3 = keras_batchnormalization_relu(conv3)
    print "conv3 shape:", conv3.shape
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    print "pool3 shape:", pool3.shape

    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(pool3)
    conv4 = keras_batchnormalization_relu(conv4)
    print "conv4 shape:", conv4.shape
    conv4 = Conv3D(256, (3, 3, 3), dilation_rate=(2, 2, 2), activation='relu', padding='same')(conv4)
    conv4 = keras_batchnormalization_relu(conv4)
    print "conv4 shape:", conv4.shape

    deconv6 = Deconvolution3D(128, 2, output_shape=(None, 40, 60, 24, 128),
                         strides=(2, 2, 2),
                         padding='valid',
                         input_shape=(32, 32, 6, 256), name='deconv6')(conv4)
    deconv6 = keras_batchnormalization_relu(deconv6)
    deconv6 = Conv3D(128, 3, activation='relu', padding='same')(deconv6)
    deconv6 = keras_batchnormalization_relu(deconv6)
    conv_conv4 = Conv3D(128, 3, activation='relu', padding='same')(conv3)
    conv_conv4 = keras_batchnormalization_relu(conv_conv4)
    up6 = concatenate([deconv6, conv_conv4],axis=-1)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = keras_batchnormalization_relu(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv6)
    conv6 = keras_batchnormalization_relu(conv6)


    deconv7 = Deconvolution3D(64, 2, output_shape=(None, 80, 120, 48, 64),
                              strides=(2, 2, 2),
                              padding='valid',
                              input_shape=(64, 64, 12, 128), name='deconv7')(conv6)
    deconv7 = keras_batchnormalization_relu(deconv7)
    deconv7 = Conv3D(64, 3, activation='relu', padding='same')(deconv7)
    deconv7 = keras_batchnormalization_relu(deconv7)
    conv_conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv2)
    conv_conv3 = keras_batchnormalization_relu(conv_conv3)
    up7 = concatenate([deconv7, conv_conv3],axis=-1)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = keras_batchnormalization_relu(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv7)
    conv7 = keras_batchnormalization_relu(conv7)


    deconv8 = Deconvolution3D(32, 2, output_shape=(None, 160, 240, 96, 32),
                              strides=(2, 2, 2),
                              padding='valid',
                              input_shape=(64, 64, 24, 64), name='deconv8')(conv7)
    deconv8 = keras_batchnormalization_relu(deconv8)
    deconv8 = Conv3D(32, 3, activation='relu', padding='same')(deconv8)
    deconv8 = keras_batchnormalization_relu(deconv8)
    conv_conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv1)
    conv_conv2 = keras_batchnormalization_relu(conv_conv2)
    up8 = concatenate([deconv8, conv_conv2], axis=-1)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = keras_batchnormalization_relu(conv8)
    conv8 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv8)
    conv8 = keras_batchnormalization_relu(conv8)


    conv10 = Conv3D(classes, (1, 1, 1), activation='relu')(conv8)
    conv10 = keras_batchnormalization_relu(conv10)
    act = Conv3D(1, 1, activation='sigmoid')(conv10)
    # act = Activation('sigmoid')(conv10)
    # model = Model(inputs=inputs, outputs=act)
    #
    # # model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=Adadelta(), loss=dice_coef_loss, metrics=[dice_coef])

    return act
