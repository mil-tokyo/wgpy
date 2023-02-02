import sys
import numpy as np
from wgpy_backends.runtime.ndarray import ndarray

def array(x):
    return asarray(np.array(x))

def zeros(*args, **kwargs):
    return asarray(np.zeros(*args, **kwargs))

def ones(*args, **kwargs):
    return asarray(np.ones(*args, **kwargs))

def ones_like(x):
    # TODO GPU native
    assert isinstance(x, ndarray)
    return asarray(np.ones(x.shape, x.dtype))

def zeros_like(x):
    # TODO GPU native
    assert isinstance(x, ndarray)
    return asarray(np.zeros(x.shape, x.dtype))

def empty(shape, dtype):
    ary = ndarray(shape, dtype=np.dtype(dtype))
    return ary

def eye(*args, **kwargs):
    # TODO GPU native
    return asarray(np.eye(*args, **kwargs))

def asarray(x) -> ndarray:
    if isinstance(x, ndarray):
        return x
    if np.isscalar(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    ary = ndarray(x.shape, x.dtype)
    ary.set(x)
    return ary

def asnumpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, ndarray):
        return x.get()
    else:
        return np.asarray(x)

def to_cpu(x):
    return asnumpy(x)

def to_gpu(x):
    return asarray(x)

def get_array_module(*args):
    for x in args:
        if isinstance(x, ndarray):
            current_module = sys.modules['wgpy'] # cupy may be needed in some case?
            return current_module
    else:
        return np

__all__ = ['array', 'empty', 'zeros', 'ones', 'eye', 'ones_like', 'zeros_like', 'asarray', 'asnumpy', 'to_cpu', 'to_gpu', 'get_array_module']
