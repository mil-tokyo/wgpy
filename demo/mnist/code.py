# Training loop
# based on Chainer's tutorial
# https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

import micropip

await micropip.install("/lib/chainer-5.4.0-py3-none-any.whl")
from pyodide.http import pyfetch

import json
from js import pythonIO
import numpy as np
import chainer
from chainer import training
from chainer import iterators, optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


response = await pyfetch(
    "/lib/mnist-subset-5k.zip"
)  # mnist.zip contains train.npz, test.npz
await response.unpack_archive(
    extract_dir="/home/pyodide/.chainer/dataset/pfnet/chainer/mnist", format="zip"
)

chainer.print_runtime_info()

from chainer.datasets import mnist

train, test = mnist.get_mnist(ndim=3)

batchsize = int(pythonIO.config.batchSize)

np.random.seed(1)

from chainer.datasets import split_dataset_random

gpu_id = -1  # cpu
try:
    import cupy as cp

    gpu_id = 0  # gpu
except:
    pass

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, False)


class MLP(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CNN(Chain):
    def __init__(self, ch=64, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, ch, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1)
            self.bn3 = L.BatchNormalization(ch)
            self.linear = L.Linear(None, n_out)

    def forward(self, x):
        h = x
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pooling_2d(h, ksize=2)
        h = F.relu(self.bn3(self.conv3(h)))
        # h = F.max_pooling_2d(h, ksize=2)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.linear(h)
        return h


model = {"MLP": MLP, "CNN": CNN}[pythonIO.config.model]()

if gpu_id >= 0:
    model.to_gpu(gpu_id)

max_epoch = 10

from chainer import optimizers

# Choose an optimizer algorithm
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

# Give the optimizer a reference to the model so that it
# can locate the model's parameters.
optimizer.setup(model)

import numpy as np
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

max_epoch = 10

i = 0

while train_iter.epoch < max_epoch:
    i += 1

    # ---------- One iteration of the training loop ----------
    train_batch = train_iter.next()
    image_train, target_train = concat_examples(train_batch, gpu_id)

    # Calculate the prediction of the network
    prediction_train = model(image_train)

    # Calculate the loss with softmax_cross_entropy
    loss = F.softmax_cross_entropy(prediction_train, target_train)

    # Calculate the gradients in the network
    model.cleargrads()
    loss.backward()

    # Update all the trainable parameters
    optimizer.update()
    # --------------------- until here ---------------------

    # Check the validation accuracy of prediction after every epoch
    if i % 10 == 0:

        # Display the training loss
        training_loss = float(to_cpu(loss.array))
        print(
            "epoch:{:02d} train_loss:{:.04f} ".format(train_iter.epoch, training_loss),
            end="",
        )

        test_losses = []
        test_accuracies = []
        with chainer.using_config("train", False), chainer.using_config(
            "enable_backprop", False
        ):
            is_first_batch = True
            for test_batch in test_iter:
                image_test, target_test = concat_examples(test_batch, gpu_id)

                # Forward the test data
                prediction_test = model(image_test)

                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, target_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

                if is_first_batch:
                    # prediction of first 10 samples for visualization
                    predicted_labels = np.argmax(to_cpu(prediction_test.array), axis=1)[
                        :10
                    ].tolist()
                    correct_labels = to_cpu(target_test)[:10].tolist()
                    is_first_batch = False

            test_iter.reset()

            test_loss = float(np.mean(test_losses))
            test_accuracy = float(np.mean(test_accuracies))

            print(
                "val_loss:{:.04f} val_accuracy:{:.04f}".format(test_loss, test_accuracy)
            )

            pythonIO.trainingProgress(
                json.dumps(
                    dict(
                        iterations=i,
                        trainingLoss=training_loss,
                        testLoss=test_loss,
                        testAccuracy=test_accuracy,
                        predictedLabels=predicted_labels,
                        correctLabels=correct_labels,
                    )
                )
            )
