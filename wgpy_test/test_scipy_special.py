import numpy as np
import wgpy as cp
import scipy.special
import cupyx.scipy


def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)


def test_erf():
    n1 = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    n1g = cp.asarray(n1)
    n2 = scipy.special.erf(n1)
    t2 = cupyx.scipy.special.erf(n1g)
    allclose(n2, cp.asnumpy(t2))


def test_get_array_module():
    n1 = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
    n1g = cp.asarray(n1)

    scipy_cpu = cupyx.scipy.get_array_module(n1)
    scipy_gpu = cupyx.scipy.get_array_module(n1g)
    assert scipy_cpu is scipy
    assert scipy_gpu is cupyx.scipy
    n3 = scipy_cpu.special.erf(n1)
    t3 = scipy_gpu.special.erf(n1g)
    allclose(n3, cp.asnumpy(t3))
