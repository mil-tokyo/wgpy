import math
import pytest
import numpy as np
import wgpy as cp

try:
    from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
except ImportError as ex:
    pytest.skip("WebGPU is not available", allow_module_level=True)


def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)


def test_elementwise_1():
    kernel = ElementwiseKernel(
        in_params="f32 x, f32 y",
        out_params="f32 z",
        operation="z = x - y",
        name="test_elementwise_1",
        uniforms="",
    )

    x = np.array([1.0, 2.0, -3.0, 2.5], dtype=np.float32)
    y = np.array([0.25, 10.0, 9.0, 1.0], dtype=np.float32)
    z_gpu = kernel(cp.asarray(x), cp.asarray(y))
    allclose(x - y, cp.asnumpy(z_gpu))

    # broadcasting
    x = np.array([[1.0, 2.0, -3.0, 2.5]], dtype=np.float32)
    y = np.array([[0.25], [10.0], [9.0], [1.0]], dtype=np.float32)
    z_gpu = kernel(cp.asarray(x), cp.asarray(y))
    allclose(x - y, cp.asnumpy(z_gpu))


def test_elementwise_raw_1():
    kernel = ElementwiseKernel(
        in_params="raw T x, i32 p",
        out_params="T y",
        operation="y = x(p)",
        name="test_elementwise_raw_1",
        uniforms="",
    )

    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    p = np.array([23, 20, 15, 0], dtype=np.int32)
    # raw T x is accessed as if x is flattened array
    y_gpu = kernel(cp.asarray(x), cp.asarray(p))
    allclose(x.flatten()[p], cp.asnumpy(y_gpu))

    x = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    p = np.array([23, 20, 15, 0], dtype=np.int32)
    # raw T x is accessed as if x is "c-contiguous" flattened array even if it has different strides
    y_gpu = kernel(cp.asarray(x).T, cp.asarray(p))
    allclose(x.T.flatten()[p], cp.asnumpy(y_gpu))


# TODO: user-defined uniform
