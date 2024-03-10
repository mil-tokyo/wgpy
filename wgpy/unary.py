from wgpy_backends.runtime.ndarray import ndarray


def exp(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.ufunc.exp(x, **kwargs)


def log(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.ufunc.log(x, **kwargs)


def tanh(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.ufunc.tanh(x, **kwargs)


def reciprocal(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.ufunc.reciprocal(x, **kwargs)


def sqrt(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.ufunc.sqrt(x, **kwargs)


__all__ = ["exp", "log", "tanh", "reciprocal", "sqrt"]
