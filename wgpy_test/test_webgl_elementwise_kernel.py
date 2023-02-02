import math
import pytest
import numpy as np
import wgpy as cp
from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

def test_elementwise_raw_1():
    kernel = ElementwiseKernel(
        in_params="raw T x, int p",
        out_params="T y",
        operation="y = x(p)",
        name="test_elementwise_raw_1",
        uniforms="",
    )

    x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
    p = np.array([23, 20, 15, 0], dtype=np.int32)
    # raw T x is accessed as if x is flattened array
    y_gpu = kernel(cp.asarray(x), cp.asarray(p))
    allclose(x.flatten()[p], cp.asnumpy(y_gpu))

    x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
    p = np.array([23, 20, 15, 0], dtype=np.int32)
    # raw T x is accessed as if x is "c-contiguous" flattened array even if it has different strides
    y_gpu = kernel(cp.asarray(x).T, cp.asarray(p))
    allclose(x.T.flatten()[p], cp.asnumpy(y_gpu))
