from wgpy_backends.runtime.ndarray import ndarray

def sum(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.sum(x, **kwargs)

def max(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.max(x, **kwargs)

def min(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.min(x, **kwargs)

def mean(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.mean(x, **kwargs)

def var(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.var(x, **kwargs)

__all__ = ['sum','max','min','mean','var']
