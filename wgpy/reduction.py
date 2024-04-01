from wgpy_backends.runtime.ndarray import ndarray


def sum(x: ndarray, axis=None, dtype=None, out=None, keepdims=False) -> ndarray:
    return x.array_func.reduction.sum(
        x, axis=axis, dtype=dtype, out=out, keepdims=keepdims
    )


def max(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.max(x, **kwargs)


def min(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.min(x, **kwargs)


def mean(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.mean(x, **kwargs)


def var(x: ndarray, **kwargs) -> ndarray:
    return x.array_func.reduction.var(x, **kwargs)


__all__ = ["sum", "max", "min", "mean", "var"]
