#!/usr/bin/env python3
import sys
import os
import shutil
import numpy as np
from PIL import Image

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


def extract_reorder_test(test_data, size):
    test_x = test_data["x"]
    test_y = test_data["y"]
    # test_y contains label from 0 to 9 in random order
    # reorder label to be 0, 1, ..., 9, 0, 1, ..., 9, ...
    idxs_for_label = {}
    for label in range(10):
        idxs_for_label[label] = np.flatnonzero(test_y == label)
    extract_idxs = []
    for i in range(size):
        extract_idxs.append(idxs_for_label[i % 10][i // 10])
    test_y = test_y[extract_idxs]
    test_x = test_x[extract_idxs]
    return {"x": test_x, "y": test_y}


def extract_train(train_data, size):
    train_x = train_data["x"]
    train_y = train_data["y"]
    return {"x": train_x[:size], "y": train_y[:size]}


# create subset
tmp_dir = "lib/tmp_mnist"
os.makedirs(tmp_dir)
train_data = np.load(dataset_dir + "/train.npz")
train_data_subset = extract_train(train_data, 5000)
np.savez_compressed(tmp_dir + "/train.npz", **train_data_subset)
test_data = np.load(dataset_dir + "/test.npz")
test_data_subset = extract_reorder_test(test_data, 100)
np.savez_compressed(tmp_dir + "/test.npz", **test_data_subset)

shutil.make_archive("lib/mnist-subset-5k", "zip", tmp_dir)

# save first 10 images as PNG
os.makedirs("lib/mnist-test", exist_ok=True)
for i in range(10):
    img = Image.fromarray(test_data_subset["x"][i].reshape(28, 28))
    img.save(f"lib/mnist-test/mnist-{i}.png")

print("lib/mnist-subset-5k.zip is created")
# remove temp files
shutil.rmtree(tmp_dir)
