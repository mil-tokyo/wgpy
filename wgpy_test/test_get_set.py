import math
import numpy as np
import wgpy as cp
from wgpy_backends.webgl.texture import WebGLArrayTextureShape, WebGL2RenderingContext, enqueue_default_texture_shape

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

def test_get_one_1d():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = t1[2] # ndarray of shape=(), not np.float32 or float
    n2 = cp.asnumpy(t2)
    allclose(n1[2], n2)

def test_get_slice_1d():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = t1[1:3]
    n2 = cp.asnumpy(t2)
    allclose(n1[1:3], n2)

def test_get_slice_3d():
    n1 = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    t1 = cp.asarray(n1)
    t2 = t1[1, np.newaxis, 3:0:-1]
    n2 = cp.asnumpy(t2)
    allclose(n1[1, np.newaxis, 3:0:-1], n2)

def test_get_adv_list():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = t1[[0, 3]]
    n2 = cp.asnumpy(t2)
    allclose(n1[[0, 3]], n2)

def test_get_adv_binary():
    n1 = np.array([5, 1, -3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    # TODO: cast in ufunc.gt
    #t2 = t1[t1 > 0]
    t2 = t1[t1 > 0.0]
    n2 = cp.asnumpy(t2)
    allclose(n1[n1 > 0], n2)

def test_set_slice_1d():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    nx = np.array([20, 30], dtype=np.float32)
    t1 = cp.asarray(n1)
    t1[1:3] = cp.asarray(nx)
    n2 = cp.asnumpy(t1)
    n1[1:3] = nx
    allclose(n1, n2)

def test_set_slice_view():
    n1 = np.arange(2*3*4, dtype=np.float32).reshape(2,3,4)
    nx = np.array([20, 30], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = t1[:, 2, 2:0:-1] # (2, 2)
    t2[1, :] = nx
    nret = cp.asnumpy(t1)
    n2 = n1[:, 2, 2:0:-1]
    n2[1, :] = nx
    allclose(n1, nret)

def test_set_adv_list():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    t1[[0, 3]] = np.array([10, 40])
    n1[[0, 3]] = np.array([10, 40])
    allclose(n1, cp.asnumpy(t1))

def test_set_adv_binary():
    n1 = np.array([5, 1, -3, 4], dtype=np.float32)
    t1 = cp.asarray(n1)
    t1[t1 > 0.0] = np.array([50, 10, 40])
    n1[n1 > 0] = np.array([50, 10, 40])
    allclose(n1, cp.asnumpy(t1))
