import pytest
import numpy as np
import cupy as cp  # use cupy-compatible interface
import chainer
from chainer import Chain, Sequential, optimizers
import chainer.functions as F
import chainer.links as L


def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)


@pytest.fixture(autouse=True)
def setup_teardown():
    np.random.seed(1)
    yield


def forward_backward_link(link, input_shape_or_array, link_attrs, forward_only=False):
    link.cleargrads()

    if isinstance(input_shape_or_array, tuple):
        data = np.random.randn(*input_shape_or_array).astype(np.float32)
    else:
        data = input_shape_or_array
    data_cpu = chainer.Variable(data)
    out_cpu = link(data_cpu)
    if not forward_only:
        rand_pattern = np.random.randn(*out_cpu.shape).astype(np.float32)
        loss = F.mean_squared_error(out_cpu, rand_pattern)
        loss.backward()
        grad_cpu_link_attrs = {}
        for link_attr in link_attrs:
            grad_cpu_link_attrs[link_attr] = getattr(link, link_attr).grad
        grad_cpu_input = data_cpu.grad

    link.to_gpu(0)
    link.cleargrads()
    data_gpu = chainer.Variable(cp.asarray(data))
    out_gpu = link(data_gpu)
    if not forward_only:
        loss = F.mean_squared_error(out_gpu, cp.asarray(rand_pattern))
        loss.backward()
        grad_gpu_link_attrs = {}
        for link_attr in link_attrs:
            grad_gpu_link_attrs[link_attr] = getattr(link, link_attr).grad
        grad_gpu_input = data_gpu.grad

    allclose(out_cpu.array, cp.asnumpy(out_gpu.array))
    if not forward_only:
        allclose(grad_cpu_input, cp.asnumpy(grad_gpu_input))
        for link_attr in link_attrs:
            allclose(
                grad_cpu_link_attrs[link_attr],
                cp.asnumpy(grad_gpu_link_attrs[link_attr]),
            )


def test_linear():
    forward_backward_link(
        L.Linear(8, 4, initial_bias=np.random.randn(4).astype(np.float32)),
        (2, 8),
        ["W", "b"],
    )


def test_relu():
    forward_backward_link(Sequential(F.relu), (2, 8), [])


def test_sigmoid():
    forward_backward_link(
        Sequential(F.sigmoid),
        np.array([0.0, 0.5, 1.0, 100.0, -100.0], dtype=np.float32),
        [],
    )


def test_batch_normalization():
    input_shape = (10, 3, 7, 8)
    input_batches = [np.random.rand(*input_shape).astype(np.float32) for _ in range(4)]
    rand_patterns = [np.random.rand(*input_shape).astype(np.float32) for _ in range(4)]
    val_batch = np.random.rand(*input_shape).astype(np.float32)

    def do_train(use_gpu):
        if use_gpu:
            c = lambda v: v.to_cpu()
            g = lambda v: cp.to_gpu(v)
        else:
            c = lambda v: v
            g = lambda v: v

        link = L.BatchNormalization(input_shape[1])
        if use_gpu:
            link.to_gpu()
        optimizer = optimizers.MomentumSGD()
        optimizer.setup(link)

        with chainer.using_config("train", True):
            for x, t in zip(input_batches, rand_patterns):
                x_var = chainer.Variable(g(x))
                y = link(x_var)
                loss = F.mean_squared_error(y, g(t))
                link.cleargrads()
                loss.backward()
                optimizer.update()
                last_y_data = y.data
                last_x_grad = x_var.grad

        with chainer.using_config("train", False):
            val_y = link(g(val_batch))

        return (
            c(link.gamma.data),
            c(link.beta.data),
            c(link.avg_mean),
            c(link.avg_var),
            c(last_y_data),
            c(last_x_grad),
            c(val_y.data),
        )

    cpu_result = do_train(False)
    gpu_result = do_train(True)
    for cr, gr in zip(cpu_result, gpu_result):
        allclose(cr, gr)


def test_max_pooling_2d_1():
    forward_backward_link(
        Sequential(lambda x: F.max_pooling_2d(x, ksize=3, stride=2, pad=1)),
        (2, 3, 7, 8),
        [],
    )


def test_average_pooling_2d_1():
    forward_backward_link(
        Sequential(lambda x: F.average_pooling_2d(x, ksize=3, stride=2, pad=1)),
        (2, 3, 7, 8),
        [],
    )


def test_convolution_2d_1():
    # forward: im2col, tensordot()
    forward_backward_link(
        L.Convolution2D(
            in_channels=3,
            out_channels=4,
            ksize=3,
            stride=1,
            pad=1,
            initial_bias=np.random.randn(4).astype(np.float32),
        ),
        (2, 3, 7, 8),
        ["W", "b"],
    )

    forward_backward_link(
        L.Convolution2D(
            in_channels=1,
            out_channels=8,
            ksize=3,
            stride=1,
            pad=1,
            initial_bias=np.random.randn(8).astype(np.float32),
        ),
        (2, 1, 28, 28),
        ["W", "b"],
    )


def test_cnn_1():
    class CNN(Chain):
        def __init__(self, ch=8, n_out=10):
            super().__init__()
            with self.init_scope():
                self.conv1 = L.Convolution2D(1, ch, ksize=3, stride=1, pad=1)
                self.conv2 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1)
                self.conv3 = L.Convolution2D(ch, ch, ksize=3, stride=1, pad=1)
                self.linear = L.Linear(None, n_out)

        def forward(self, x):
            h = x
            h = F.relu(self.conv1(h))
            h = F.max_pooling_2d(h, ksize=2)
            h = F.relu(self.conv2(h))
            h = F.max_pooling_2d(h, ksize=2)
            h = F.relu(self.conv3(h))
            h = F.max_pooling_2d(h, ksize=2)
            h = self.linear(h)
            return h

    forward_backward_link(CNN(), (2, 1, 28, 28), [])
