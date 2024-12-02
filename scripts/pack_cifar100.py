#!/usr/bin/env python3
import sys
import os
import shutil
import zipfile
import numpy as np

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

# create subset
fullset = np.load(dataset_dir + "/cifar-100.npz")
train_limit = 5000
test_limit = 1000
np.savez(
    "lib/cifar-100-subset-5k.npz",
    train_x=fullset["train_x"][:train_limit],
    train_y=fullset["train_y"][:train_limit],
    test_x=fullset["test_x"][:test_limit],
    test_y=fullset["test_y"][:test_limit],
)

with zipfile.ZipFile("lib/cifar100-subset-5k.zip", "w") as zf:
    zf.write("cifar-100-subset-5k.npz", arcname="cifar-100.npz")

print("lib/cifar100.zip is created")
print("lib/cifar100-subset-5k.zip is created")
# remove temp files
os.remove("lib/cifar-100-subset-5k.npz")
