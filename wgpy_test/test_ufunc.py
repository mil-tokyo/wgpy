import math
import pytest
import numpy as np
import wgpy as cp
from wgpy_backends.webgl.texture import WebGLArrayTextureShape, WebGL2RenderingContext, enqueue_default_texture_shape

def allclose(expected, actual):
    np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

def test_add():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 + t2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_iadd():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    t1 = cp.asarray(n1)
    t1v = t1[:]
    t2 = cp.asarray(n2)
    t1 += t2
    # inplace operation: t1v should follow update of t1
    # inplace update of view is not yet implemented (t1v+=1)
    allclose(n1 + n2, cp.asnumpy(t1v))

def test_iadd_view():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5], dtype=np.float32)
    t1 = cp.asarray(n1)
    t1v = t1[1:3]
    # inplace update of view is not yet implemented
    with pytest.raises(NotImplementedError):
        t1v += n2
    
    # n1[1:3] += n2
    # allclose(n1, cp.asnumpy(t1))

def test_add_int32():
    n1 = np.array([1, 2, 3, 2147483647], dtype=np.int32)
    n2 = np.array([2, 100, 200, -10], dtype=np.int32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 + t2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_add_broadcast_1():
    n1 = np.array([[1], [2]], dtype=np.float32)
    n2 = np.array([-0.5, 1.5], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 + t2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_add_scalar_1():
    n1 = np.array([[1], [2]], dtype=np.int32)
    n2 = 5
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 + t2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_add_scalar_2():
    n1 = np.array([[1.3], [2.5]], dtype=np.float32)
    n2 = 3.3
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 + t2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_add_scalar_3():
    n1 = np.array([[1.3], [2.5]], dtype=np.float32)
    n2 = 4.4
    t1 = cp.asarray(n1)
    t3 = t1 + n2
    n3 = cp.asnumpy(t3)
    allclose(n1 + n2, n3)

def test_radd():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = n2 + t1 # numpy + cupy
    allclose(n2 + n1, cp.asnumpy(t2))

def test_radd_scalar():
    n1 = np.array([[1.3], [2.5]], dtype=np.float32)
    n2 = 3.3
    t1 = cp.asarray(n1)
    t2 = n2 + t1
    allclose(n2 + n1, cp.asnumpy(t2))

def test_sub_irregular_texture_1():
    enqueue_default_texture_shape(
        WebGLArrayTextureShape(height=1, width=4, depth=1, dim='2D'),
        WebGLArrayTextureShape(height=2, width=2, depth=1, dim='2D'),
        WebGLArrayTextureShape(height=2, width=2, depth=1, dim='2D'),
    )
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.2], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 - t2
    n3 = cp.asnumpy(t3)
    allclose(n1 - n2, n3)


def test_sub_irregular_texture_2():
    shape = (15, 823) # 12345
    size = shape[0] * shape[1]
    enqueue_default_texture_shape(
        WebGLArrayTextureShape(height=int(math.ceil(size / 15)), width=15, depth=1, dim='2D'),
        WebGLArrayTextureShape(height=int(math.ceil(size / 765)), width=765, depth=1, dim='2D'),
        WebGLArrayTextureShape(height=int(math.ceil(size / 16)), width=16, depth=1, dim='2D'),
    )
    np.random.seed(1)
    n1 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    n2 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 - t2
    n3 = cp.asnumpy(t3)
    allclose(n1 - n2, n3)

def test_sub_irregular_texture_3():
    shape = (15, 823) # 12345
    size = shape[0] * shape[1]
    enqueue_default_texture_shape(
        WebGLArrayTextureShape(height=int(math.ceil(size / 30)), width=15, depth=2, dim='2DArray'),
        WebGLArrayTextureShape(height=int(math.ceil(size / 765)), width=765, depth=1, dim='2D'),
        WebGLArrayTextureShape(height=int(math.ceil(size / 48)), width=16, depth=3, dim='2DArray'),
    )
    np.random.seed(1)
    n1 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    n2 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 - t2
    n3 = cp.asnumpy(t3)
    allclose(n1 - n2, n3)


def test_sub_irregular_texture_4():
    shape = (15, 823) # 12345
    size = shape[0] * shape[1]
    enqueue_default_texture_shape(
        WebGLArrayTextureShape(height=int(math.ceil(size / (15*4))), width=15, depth=1, dim='2D', internal_format=WebGL2RenderingContext.RGBA16F, format=WebGL2RenderingContext.RGBA, type=WebGL2RenderingContext.HALF_FLOAT),
        WebGLArrayTextureShape(height=int(math.ceil(size / (765*4))), width=765, depth=1, dim='2D', internal_format=WebGL2RenderingContext.RGBA16F, format=WebGL2RenderingContext.RGBA, type=WebGL2RenderingContext.HALF_FLOAT),
        WebGLArrayTextureShape(height=int(math.ceil(size / (48*4))), width=16, depth=3, dim='2DArray',internal_format=WebGL2RenderingContext.RGBA16F, format=WebGL2RenderingContext.RGBA, type=WebGL2RenderingContext.HALF_FLOAT),
    )
    np.random.seed(1)
    n1 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    n2 = np.random.randint(-4, 5, size=shape).astype(np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 - t2
    n3 = cp.asnumpy(t3)
    allclose(n1 - n2, n3)

def test_rsub():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = n2 - t1 # numpy - cupy
    allclose(n2 - n1, cp.asnumpy(t2))

def test_rsub_scalar():
    n1 = np.array([[1.3], [2.5]], dtype=np.float32)
    n2 = 3.3
    t1 = cp.asarray(n1)
    t2 = n2 - t1
    allclose(n2 - n1, cp.asnumpy(t2))

def test_gt():
    n1 = np.array([1, 2, 3, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.0], dtype=np.float32)
    tr = cp.asarray(n1) > cp.asarray(n2)
    assert tr.dtype == np.bool_
    allclose(n1 > n2, cp.asnumpy(tr))

def test_pos():
    n1 = np.array([-1.0, 0.0, 2.5, 123.45], dtype=np.float32)
    tr = +cp.asarray(n1)
    allclose(+n1, cp.asnumpy(tr))

def test_neg():
    n1 = np.array([-1.0, 0.0, 2.5, 123.45], dtype=np.float32)
    tr = -cp.asarray(n1)
    allclose(-n1, cp.asnumpy(tr))

    n1 = np.array([-1, 0, 1, 2147483647], dtype=np.int32)
    tr = -cp.asarray(n1)
    allclose(-n1, cp.asnumpy(tr))

    n1 = np.array([0, 1, 2, 255], dtype=np.uint8)
    tr = -cp.asarray(n1)
    allclose(-n1, cp.asnumpy(tr))

    # not supported by numpy
    # n1 = np.array([False, True], dtype=np.bool_)
    # tr = -cp.asarray(n1)
    # allclose(-n1, cp.asnumpy(tr))

def test_abs():
    n1 = np.array([-1.0, 0.0, 2.5, 123.45], dtype=np.float32)
    tr = abs(cp.asarray(n1))
    allclose(abs(n1), cp.asnumpy(tr))

    n1 = np.array([-1, 0, 1, -2147483647], dtype=np.int32)
    tr = abs(cp.asarray(n1))
    allclose(abs(n1), cp.asnumpy(tr))

    n1 = np.array([0, 1, 2, 255], dtype=np.uint8)
    tr = abs(cp.asarray(n1))
    allclose(abs(n1), cp.asnumpy(tr))

    n1 = np.array([False, True], dtype=np.bool_)
    tr = abs(cp.asarray(n1))
    allclose(abs(n1), cp.asnumpy(tr))

def test_invert():
    n1 = np.array([-1, 0, 1, -2147483647], dtype=np.int32)
    tr = ~cp.asarray(n1)
    allclose(~n1, cp.asnumpy(tr))

    n1 = np.array([0, 1, 2, 255], dtype=np.uint8)
    tr = ~cp.asarray(n1)
    allclose(~n1, cp.asnumpy(tr))

    n1 = np.array([False, True], dtype=np.bool_)
    tr = ~cp.asarray(n1)
    allclose(~n1, cp.asnumpy(tr))

def test_exp():
    n1 = np.array([-1.0, 0.0, 2.5, 0.1], dtype=np.float32)
    tr = cp.exp(cp.asarray(n1))
    allclose(np.exp(n1), cp.asnumpy(tr))
    
    n1 = np.array([-1.0, 0.0, 2.5, 0.1], dtype=np.float32)
    out = cp.empty(n1.shape, dtype=n1.dtype)
    cp.exp(cp.asarray(n1), out=out)
    allclose(np.exp(n1), cp.asnumpy(out))

def test_log():
    n1 = np.array([1.0, 2.5, 0.1], dtype=np.float32)
    tr = cp.log(cp.asarray(n1))
    allclose(np.log(n1), cp.asnumpy(tr))

def test_tanh():
    n1 = np.array([-1.0, 0.0, 2.5, 0.1], dtype=np.float32)
    tr = cp.tanh(cp.asarray(n1))
    allclose(np.tanh(n1), cp.asnumpy(tr))

def test_maximum():
    # currently, nan is not handled properly
    n1 = np.array([1, -1.0, -np.inf, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, np.inf], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.maximum(t1, t2)
    allclose(np.maximum(n1, n2), cp.asnumpy(t3))

    t4 = cp.maximum(t1, 0)
    allclose(np.maximum(n1, 0), cp.asnumpy(t4))

def test_fmax():
    # currently, nan is not handled properly
    n1 = np.array([1, -1.0, -np.inf, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, np.inf], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.fmax(t1, t2)
    allclose(np.fmax(n1, n2), cp.asnumpy(t3))

    t4 = cp.fmax(t1, 0)
    allclose(np.fmax(n1, 0), cp.asnumpy(t4))

def test_minimum():
    # currently, nan is not handled properly
    n1 = np.array([1, -1.0, -np.inf, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, np.inf], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.minimum(t1, t2)
    allclose(np.minimum(n1, n2), cp.asnumpy(t3))

    t4 = cp.minimum(t1, 0)
    allclose(np.minimum(n1, 0), cp.asnumpy(t4))

def test_fmin():
    # currently, nan is not handled properly
    n1 = np.array([1, -1.0, -np.inf, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, np.inf], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.fmin(t1, t2)
    allclose(np.fmin(n1, n2), cp.asnumpy(t3))

    t4 = cp.fmin(t1, 0)
    allclose(np.fmin(n1, 0), cp.asnumpy(t4))

def test_divide():
    n1 = np.array([1, -1.0, 3.0, 4], dtype=np.float32)
    n2 = np.array([-0.5, 1.5, 2.0, 0.5], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.divide(t1, t2)
    allclose(np.divide(n1, n2), cp.asnumpy(t3))

    t4 = cp.divide(t1, 0.5)
    allclose(np.divide(n1, 0.5), cp.asnumpy(t4))

def test_divide_int():
    n1 = np.array([10, 20, -20, 5], dtype=np.int32)
    n2 = np.array([3, 1, -3, -2], dtype=np.int32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = cp.divide(t1, t2) # truediv: float is returned
    allclose(np.divide(n1, n2), cp.asnumpy(t3))

    t4 = cp.divide(t1, 2)
    allclose(np.divide(n1, 2), cp.asnumpy(t4))
