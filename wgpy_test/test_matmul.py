import math
import numpy as np
import wgpy as cp

def allclose(expected, actual, rtol=1e-2, atol=1e-2):
    np.testing.assert_allclose(expected, actual, rtol=rtol, atol=atol)

def test_matmul():
    n1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    n2 = np.array([[-0.5, 1.5], [2.0, 0.0]], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 @ t2
    n3 = cp.asnumpy(t3)
    allclose(n1 @ n2, n3)

def test_matmul_multi_shape():
    np.random.seed(1)
    for m in [1, 8, 17, 65]:
        for n in [1, 8, 17, 65]:
            for k in [1, 8, 17, 65]:
                n1 = np.random.randint(-4, 5, size=(m, k)).astype(np.float32)
                n2 = np.random.randint(-4, 5, size=(k, n)).astype(np.float32)
                n3actual = cp.asnumpy(cp.asarray(n1) @ cp.asarray(n2))
                n3expect = n1 @ n2
                allclose(n3actual, n3expect)

                out = cp.empty(n3expect.shape, dtype=np.float32)
                cp.matmul(cp.asarray(n1), cp.asarray(n2), out=out)
                allclose(cp.asnumpy(out), n3expect)
