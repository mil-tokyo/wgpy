# Training loop
# based on Chainer's tutorial
# https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--out", type=str, default="resnet_result")
args = parser.parse_args()

batchsize = args.batchsize

np.random.seed(1)

from chainer.datasets import split_dataset_random

gpu_id = args.gpu  # cpu
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

max_epoch = args.epoch

# Wrap your model by Classifier and include the process of loss calculation within your model.
# Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
model = L.Classifier(model)

# selection of your optimizing method
optimizer = optimizers.MomentumSGD(lr=args.lr)

# Give the optimizer a reference to the model
optimizer.setup(model)

# Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# Setup a Trainer
trainer = training.Trainer(updater, (max_epoch, "epoch"), out=args.out)

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
