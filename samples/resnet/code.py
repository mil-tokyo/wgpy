# Training loop
# based on Chainer's tutorial
# https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

import micropip

await micropip.install("/lib/chainer-5.4.0-py3-none-any.whl")
await micropip.install("/samples/resnet/dist/resnet-0.0.1-py3-none-any.whl")
from pyodide.http import pyfetch

response = await pyfetch("/lib/cifar100.zip")  # mnist.zip contains train.npz, test.npz
await response.unpack_archive(
    extract_dir="/home/pyodide/.chainer/dataset/pfnet/chainer/cifar", format="zip"
)

from js import pythonIO
import numpy as np
import chainer
from chainer import training
from chainer import iterators, optimizers
import chainer.links as L
from chainer.training import extensions

chainer.print_runtime_info()
from resnet import ResNet18

from chainer.datasets import get_cifar100

train, test = get_cifar100()

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

# reduce test size to 1000
test, _ = split_dataset_random(test, 1000, seed=0)
test_iter = iterators.SerialIterator(test, batchsize, False, False)


model = ResNet18()

if gpu_id >= 0:
    model.to_gpu(gpu_id)

max_epoch = 10

# Wrap your model by Classifier and include the process of loss calculation within your model.
# Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
model = L.Classifier(model)

# selection of your optimizing method
optimizer = optimizers.MomentumSGD(lr=0.01*(batchsize/16))

# Give the optimizer a reference to the model
optimizer.setup(model)

# Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Setup a Trainer
trainer = training.Trainer(updater, (max_epoch, "epoch"), out="mnist_result")

from chainer.training import extensions

# training speed largely varies due to model selection and hardware performance. print report by constant time interval.
trainer.extend(extensions.LogReport(trigger=chainer.training.triggers.TimeTrigger(10)))
# trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
# trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(
    extensions.PrintReport(
        [
            "epoch",
            "iteration",
            "main/loss",
            "main/accuracy",
            "validation/main/loss",
            "validation/main/accuracy",
            "elapsed_time",
        ]
    )
)
# matplotlib does not work on worker
# trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
# trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

trainer.run()
try:
    import wgpy_backends.webgl

    print(wgpy_backends.webgl.get_performance_metrics())
except:
    pass
try:
    import wgpy_backends.webgpu

    print(wgpy_backends.webgpu.get_performance_metrics())
except:
    pass
