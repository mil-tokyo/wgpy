import pytest
import numpy as np
import wgpy as cp

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

@pytest.fixture(autouse=True)
def setup_teardown():
    np.random.seed(1)
    yield    

# TODO: check case when texture is 2DArray or RGBA

def test_sum():
    n1 = np.arange(4*3*2).reshape((4,3,2)).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.sum(n1), cp.asnumpy(cp.sum(t1)))
    allclose(np.sum(n1, axis=1), cp.asnumpy(cp.sum(t1, axis=1)))
    allclose(np.sum(n1, axis=(0, 2), keepdims=True), cp.asnumpy(cp.sum(t1, axis=(0, 2), keepdims=True)))

def test_max():
    n1 = np.random.rand(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.max(n1), cp.asnumpy(cp.max(t1)))
    allclose(np.max(n1, axis=1), cp.asnumpy(cp.max(t1, axis=1)))
    allclose(np.max(n1, axis=(0, 2), keepdims=True), cp.asnumpy(cp.max(t1, axis=(0, 2), keepdims=True)))

def test_min():
    n1 = np.random.rand(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.min(n1), cp.asnumpy(cp.min(t1)))
    allclose(np.min(n1, axis=1), cp.asnumpy(cp.min(t1, axis=1)))
    allclose(np.min(n1, axis=(0, 2), keepdims=True), cp.asnumpy(cp.min(t1, axis=(0, 2), keepdims=True)))

def test_mean():
    n1 = np.random.rand(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.mean(n1), cp.asnumpy(cp.mean(t1)))
    allclose(np.mean(n1, axis=1), cp.asnumpy(cp.mean(t1, axis=1)))
    allclose(np.mean(n1, axis=(0, 2), keepdims=True), cp.asnumpy(cp.mean(t1, axis=(0, 2), keepdims=True)))

def test_var():
    n1 = np.random.rand(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.var(n1), cp.asnumpy(cp.var(t1)))
    allclose(np.var(n1, axis=1), cp.asnumpy(cp.var(t1, axis=1)))
    allclose(np.var(n1, axis=(0, 2), keepdims=True), cp.asnumpy(cp.var(t1, axis=(0, 2), keepdims=True)))
