#!/usr/bin/env sh

/home/zzr/caffe/build/tools/caffe train --solver=/home/zzr/caffe/models/DenseNet-Caffe/ISIC/DenseNet_201_solver.prototxt --weights=/home/zzr/caffe/models/DenseNet-Caffe/DenseNet_201.caffemodel $SOLVERFILE 2>&1|tee /home/zzr/caffe/models/DenseNet-Caffe/ISIC/log/train_log.log
