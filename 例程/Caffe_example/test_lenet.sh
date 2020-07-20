#!/bin/sh
set -e

DATA="$HOME/caffe/examples/mnist"
BUILD="$HOME/caffe/build/tools"

$BUILD/caffe test -model $DATA/lenet_train_test.prototxt -weights $DATA/lenet_iter_10000.caffemodel -iterations 100 $@

