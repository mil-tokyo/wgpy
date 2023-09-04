import numpy as np
import wgpy as cp

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

def test_reshape():
    n1 = np.arange(4*3*2).reshape((4,3,2)).astype(np.float32)
    t1 = cp.asarray(n1)
    t2 = t1.reshape(6, 4)
    allclose(n1.reshape((6, 4)), cp.asnumpy(t2))

    t2 = cp.reshape(t1, (6, 4))
    allclose(n1.reshape((6, 4)), cp.asnumpy(t2))

    t2 = cp.reshape(t1, (-1, 4))
    allclose(n1.reshape((-1, 4)), cp.asnumpy(t2))

    t3 = t1[1:3, 2, :] # (2, 2) view, not contiguous
    t4 = t3.reshape((4,))
    n3 = n1[1:3, 2, :]
    allclose(n3.reshape((4,)), cp.asnumpy(t4))

    # TODO: when size is 0

def test_transpose():
    n1 = np.arange(4*3*2).reshape(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(n1.transpose(), cp.asnumpy(t1.transpose()))
    allclose(n1.transpose(), cp.asnumpy(cp.transpose(t1)))
    allclose(n1.transpose(), cp.asnumpy(t1.T))
    
    allclose(n1.transpose(2, 0, 1), cp.asnumpy(t1.transpose(2, 0, 1)))
    allclose(n1.transpose(2, 0, 1), cp.asnumpy(cp.transpose(t1, (2, 0, 1))))

    t3 = t1[1:3, 2, :] # (2, 2) view, not contiguous
    n3 = n1[1:3, 2, :]

    allclose(n3.transpose(), cp.asnumpy(t3.transpose()))

def test_ravel():
    n1 = np.arange(4*3*2).reshape(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(n1.ravel(), cp.asnumpy(t1.ravel()))
    allclose(n1.ravel(), cp.asnumpy(cp.ravel(t1)))
    
    t3 = t1[1:3, 2, :] # (2, 2) view, not contiguous
    n3 = n1[1:3, 2, :]

    allclose(n3.ravel(), cp.asnumpy(t3.ravel()))

def test_expand_dims():
    n1 = np.arange(4*3*2).reshape(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.expand_dims(n1, 1), cp.asnumpy(cp.expand_dims(t1, 1)))
    allclose(np.expand_dims(n1, (2, 0)), cp.asnumpy(cp.expand_dims(t1, (2, 0))))

def test_reduced_view():
    n1 = np.arange(4*3*2).reshape(4,3,2).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(n1.reshape(-1), cp.asnumpy(t1.reduced_view()))

    n1 = np.array([1.5]).astype(np.float32).reshape(())
    t1 = cp.asarray(n1)

    assert t1.reduced_view().shape == ()

def test_squeeze():
    n1 = np.arange(4*3*2).reshape(1,4,1,3,1,2,1).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(n1.squeeze(), cp.asnumpy(t1.squeeze()))
    allclose(n1.squeeze(), cp.asnumpy(cp.squeeze(t1)))

    allclose(n1.squeeze(axis=(2,4)), cp.asnumpy(t1.squeeze(axis=(2,4))))
    allclose(n1.squeeze(axis=(2,4)), cp.asnumpy(cp.squeeze(t1, axis=(2,4))))

def test_rollaxis():
    n1 = np.arange(3*4*5*6).reshape(3,4,5,6).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.rollaxis(n1, 3, 1), cp.asnumpy(cp.rollaxis(t1, 3, 1)))
    allclose(np.rollaxis(n1, 2), cp.asnumpy(cp.rollaxis(t1, 2)))
    allclose(np.rollaxis(n1, 1, 4), cp.asnumpy(cp.rollaxis(t1, 1, 4)))

def test_reduced_view():
    n1 = np.arange(2*3*4*5*6).reshape(2,3,4,5,6).astype(np.float32)
    t1 = cp.asarray(n1)
    allclose(np.reshape(n1, (n1.size,)), cp.asnumpy(t1.reduced_view()))

    allclose(
        n1.transpose((3,0,1,2,4)).reshape(5,24,6),
        cp.asnumpy(t1.transpose((3,0,1,2,4)).reduced_view())
    )

    allclose(
        n1.transpose((3,4,2,0,1)).reshape(30,4,6),
        cp.asnumpy(t1.transpose((3,4,2,0,1)).reduced_view())
    )

    # size 0 => shape (0,)
    t2 = cp.asarray(np.zeros((2, 0, 3), dtype=np.float32))
    assert t2.reduced_view().shape == (0,)

    # ndim 1 => shape (size,)
    t3 = cp.asarray(np.zeros((1,), dtype=np.float32))
    assert t3.reduced_view().shape == (1,)

    # scalar => scalar
    t4 = cp.asarray(np.zeros((), dtype=np.float32))
    assert t4.reduced_view().shape == ()

    # size 1 (not ndim 1) => scalar
    t5 = cp.asarray(np.zeros((1, 1), dtype=np.float32))
    assert t5.reduced_view().shape == ()

