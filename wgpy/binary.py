from typing import List, Tuple, Union
import numpy as np
from wgpy.construct import asarray, asnumpy
from wgpy_backends.runtime.ndarray import ndarray

def _binary(numpy_func, x, y, out=None, **kwargs) -> ndarray:
    if out is not None:
        assert isinstance(out, ndarray)
        cpu_out = np.empty(out.shape, out.dtype)
        numpy_func(asnumpy(x), asnumpy(y), out=cpu_out, **kwargs)
        out.set(cpu_out)
        return out
    return asarray(numpy_func(asnumpy(x), asnumpy(y), **kwargs))

def dot(x: ndarray, y: ndarray, out=None) -> ndarray:
    if isinstance(x, ndarray) and isinstance(y, ndarray) and x.ndim == 2 and y.ndim == 2:
        # supported case
        return x.array_func.matmul(x, y, out=out)
    return _binary(np.dot, x, y, out=out)

def matmul(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    if isinstance(x, ndarray) and isinstance(y, ndarray) and x.ndim == 2 and y.ndim == 2:
        # supported case
        return x.array_func.matmul(x, y, out=out)
    return _binary(np.matmul, x, y, out=out, **kwargs)

def tensordot(a: ndarray, b: ndarray, axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]]=2) -> ndarray:
    return a.array_func.tensordot(a, b, axes=axes)

def divide(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    x = asarray(x)
    y = asarray(y)
    return x.array_func.ufunc.truediv(x, y, out=out, **kwargs)

def maximum(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    x = asarray(x)
    y = asarray(y)
    return x.array_func.ufunc.maximum(x, y, out=out, **kwargs)

def fmax(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    x = asarray(x)
    y = asarray(y)
    return x.array_func.ufunc.fmax(x, y, out=out, **kwargs)

def minimum(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    x = asarray(x)
    y = asarray(y)
    return x.array_func.ufunc.minimum(x, y, out=out, **kwargs)

def fmin(x: ndarray, y: ndarray, out=None, **kwargs) -> ndarray:
    x = asarray(x)
    y = asarray(y)
    return x.array_func.ufunc.fmin(x, y, out=out, **kwargs)

__all__ = ['dot', 'matmul', 'tensordot', 'divide', 'maximum', 'fmax', 'minimum', 'fmin']
