from wgpy_backends.runtime.ndarray import ndarray

def get_runtime_info():
    return "wgpy cupyx mock runtime info"

def rsqrt(x: ndarray, out=None) -> ndarray:
    return x.array_func.ufunc.rsqrt(x, out=out)
