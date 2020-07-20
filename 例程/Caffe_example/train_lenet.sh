#!/bin/sh
set -e

DATA="$HOME/caffe/examples/mnist"
BUILD="$HOME/caffe/build/tools"

$BUILD/caffe train --solver=$DATA/lenet_solver.prototxt $@

