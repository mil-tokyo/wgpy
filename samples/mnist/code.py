# Training loop
# based on Chainer's tutorial
# https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

import micropip
await micropip.install('/lib/chainer-5.4.0-py3-none-any.whl')
from pyodide.http import pyfetch
response = await pyfetch('/lib/mnist.zip') # mnist.zip contains train.npz, test.npz
await response.unpack_archive(extract_dir="/home/pyodide/.chainer/dataset/pfnet/chainer/mnist", format="zip")

from js import pythonIO
import numpy as np
import chainer
from chainer import training
from chainer import iterators, optimizers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
chainer.print_runtime_info()

from chainer.datasets import mnist

train, test = mnist.get_mnist(ndim=3)

batchsize = int(pythonIO.config.batchSize)

np.random.seed(1)

from chainer.datasets import split_dataset_random

gpu_id = -1 # cpu
try:
    import cupy as cp
    gpu_id = 0 # gpu
except:
    pass

train_iter = iterators.SerialIterator(train, batchsize)

# reduce test size to 1000
test, _ = split_dataset_random(test, 1000, seed=0)
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
        #h = F.max_pooling_2d(h, ksize=2)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.linear(h)
        return h


model = {"MLP": MLP, "CNN": CNN}[pythonIO.config.model]()

if gpu_id >= 0:
    model.to_gpu(gpu_id)

max_epoch = 10

# Wrap your model by Classifier and include the process of loss calculation within your model.
# Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
model = L.Classifier(model)

# selection of your optimizing method
optimizer = optimizers.MomentumSGD()

# Give the optimizer a reference to the model
optimizer.setup(model)

# Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Setup a Trainer
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

from chainer.training import extensions

# training speed largely varies due to model selection and hardware performance. print report by constant time interval.
trainer.extend(extensions.LogReport(trigger=chainer.training.triggers.TimeTrigger(10)))
# trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
# trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
# matplotlib does not work on worker
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

trainer.run()
import wgpy_backends.webgl
print(wgpy_backends.webgl.get_performance_metrics())
