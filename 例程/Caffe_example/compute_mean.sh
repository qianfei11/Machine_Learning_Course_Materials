#!/bin/sh
set -e

DATA="$HOME/caffe/examples/mnist"
BUILD="$HOME/caffe/build/tools"

rm -rf $DATA/mean.binaryproto

$BUILD/compute_image_mean $DATA/mnist_train_lmdb $DATA/mean.binaryproto $@

