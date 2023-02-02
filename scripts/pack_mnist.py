#!/usr/bin/env python3
import sys
import os
import shutil

try:
    from chainer.datasets import get_mnist
except:
    print("Failed to load chainer. Please install it by")
    print("pip install chainer==5.4.0")
    sys.exit(1)

get_mnist()
dataset_dir = os.path.expanduser("~/.chainer/dataset/pfnet/chainer/mnist")
# there is dataset_dir/{train.npz,test.npz}
shutil.make_archive("lib/mnist", "zip", dataset_dir)

print("lib/mnist.zip is created")
