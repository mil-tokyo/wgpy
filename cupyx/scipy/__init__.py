import sys
from wgpy_backends.runtime.ndarray import ndarray
from . import special


def get_array_module(*args):
    for x in args:
        if isinstance(x, ndarray):
            current_module = sys.modules["cupyx.scipy"]
            return current_module
    else:
        import scipy

        return scipy
