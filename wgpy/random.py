from wgpy.construct import asarray
from wgpy_backends.runtime.ndarray import ndarray
import numpy as np

def randn(*args) -> ndarray:
    return asarray(np.random.randn(*args))

def normal(*args, **kwargs) -> ndarray:
    return asarray(np.random.normal(*args, **kwargs))
