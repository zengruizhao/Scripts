#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12
PRONAME=/home/zzr/Data/Skin/data

EXAMPLE=/home/zzr/Data/Skin/script_all/data
DATA=$PRONAME/train_test_all_2
TOOLS=/home/zzr/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $EXAMPLE/train_lmdb/trainset_mean.binaryproto

$TOOLS/compute_image_mean $EXAMPLE/test_lmdb \
  $EXAMPLE/test_lmdb/testset_mean.binaryproto

echo "Done."
