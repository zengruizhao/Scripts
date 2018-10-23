# coding=utf-8
import numpy as np
import sys, caffe

lmdb = '/media/zzr/SW/Skin_xml/WSI_20/lmdb/train_lmdb/trainset_mean.binaryproto'

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open(lmdb, 'rb').read()
blob.ParseFromString(bin_mean)
arr = np.array(caffe.io.blobproto_to_array(blob))
npy_mean = arr[0]
np.save('/home/zzr/Data/Skin/script_all/train_mean_lmdb.npy', npy_mean)