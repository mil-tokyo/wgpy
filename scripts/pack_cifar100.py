#!/usr/bin/env python3
import sys
import os
import shutil
import zipfile

try:
    from chainer.datasets import get_cifar100
except:
    print("Failed to load chainer. Please install it by")
    print("pip install chainer==5.4.0")
    sys.exit(1)

get_cifar100()
dataset_dir = os.path.expanduser("~/.chainer/dataset/pfnet/chainer/cifar")
# there is dataset_dir/cifar-100.npz
# cifar-10.npz may also exists and not needed.
with zipfile.ZipFile("lib/cifar100.zip", "w") as zf:
    zf.write(os.path.join(dataset_dir, "cifar-100.npz"), arcname="cifar-100.npz")

print("lib/cifar100.zip is created")
