import numpy as np
from wgpy.ndarray import ndarray
import wgpy.random as random
from wgpy.broadcast import *
from wgpy.construct import *
from wgpy.unary import *
from wgpy.binary import *
from wgpy.manipulation import *
from wgpy.reduction import *
from wgpy_backends.runtime import get_backend_name as _runtime_get_backend_name

__version__ = "1.0.0"


def isscalar(x):
    if isinstance(x, ndarray):
        # shape==() is not scalar (cupy)
        return False
    else:
        # primitives (float, int) are numpy's scalar type(np.float32) are True
        # Even if np.array(3.0).shape==(), but it is treated as False
        return np.isscalar(x)


# decorator
def memoize(for_each_device=False):
    def func(f):
        def wrapper(*args, **kwargs):
            # TODO: memoize based on args, kwargs
            return f(*args, **kwargs)

        return wrapper

    return func


# https://docs.cupy.dev/en/stable/reference/generated/cupy.fuse.html#cupy.fuse
# decorator
def fuse():
    # TODO: do kernel fusion
    def func(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper

    return func


def get_backend_name() -> str:
    """
    Returns the name of the backend currently in use.
    Possible values are 'webgpu' and 'webgl'.
    """
    return _runtime_get_backend_name()
