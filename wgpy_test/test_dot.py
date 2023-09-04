import numpy as np
import wgpy as cp

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

def test_dot_1d_1d():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    t3 = cp.dot(cp.asarray(n1), cp.asarray(n2))
    allclose(np.dot(n1, n2), cp.asnumpy(t3))

    t3 = cp.asarray(n1).dot(n2)
    allclose(np.dot(n1, n2), cp.asnumpy(t3))

def test_dot_2d_2d():
    n1 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    n2 = np.array([[4, 1], [2, 2]], dtype=np.float32)
    t3 = cp.dot(cp.asarray(n1), cp.asarray(n2))
    allclose(np.dot(n1, n2), cp.asnumpy(t3))

    t3 = cp.asarray(n1).dot(n2)
    allclose(np.dot(n1, n2), cp.asnumpy(t3))

# TODO: other case of dot
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
