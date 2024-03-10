from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from wgpy_backends.webgl.ndarray import ndarray
from wgpy_backends.webgl.reduction_kernel import ReductionKernel
from wgpy_backends.webgl.shader_util import native_scalar_type_for_dtype

reduction_kernels = {}


class ReductionExpr(NamedTuple):
    in_params: str
    out_params: str
    map_expr: str
    reduce_expr: str
    post_map_expr: str
    identity: str
    name: str
    reduce_type: Optional[str] = None
    uniforms: str = ""
    preamble: str = ""


def _get_or_create_kernel(expr: ReductionExpr) -> ReductionKernel:
    kernel = reduction_kernels.get(expr)
    if kernel is None:
        kernel = ReductionKernel(**expr._asdict())
        reduction_kernels[expr] = kernel
    return kernel


def sum(a: ndarray, axis=None, dtype=None, out=None, keepdims=False):
    if dtype is None:
        dtype = {
            np.dtype(np.bool_): np.dtype(np.int32),
            np.dtype(np.uint8): np.dtype(np.int32),
            np.dtype(np.int32): np.dtype(np.int32),
            np.dtype(np.float32): np.dtype(np.float32),
        }[a.dtype]
    else:
        dtype = np.dtype(dtype)
    scalar_type = native_scalar_type_for_dtype[dtype]
    # TODO: out_paramsのスカラー型から、ReductionKernelが出力の型を推定している。明示的に指定する手段を用意すべき。
    expr = ReductionExpr(
        in_params="T x",
        out_params=f"{scalar_type} y",
        map_expr=f"{scalar_type}(x)",
        reduce_expr="a + b",
        post_map_expr="y = a",
        identity=f"{scalar_type}(0)",
        name="sum",
    )
    return _get_or_create_kernel(expr)(a, out, axis=axis, keepdims=keepdims)


def mean(a: ndarray, axis=None, dtype=None, out=None, keepdims=False):
    if dtype is None:
        dtype = np.dtype(np.float32)
    else:
        dtype = np.dtype(dtype)
    scalar_type = native_scalar_type_for_dtype[dtype]
    if scalar_type in ["float", "int", "uint"]:
        expr = ReductionExpr(
            in_params="T x",
            out_params=f"{scalar_type} y",
            map_expr=f"{scalar_type}(x)",
            reduce_expr="a + b",
            post_map_expr=f"y = a / {scalar_type}(_in_ind_size / _out_ind_size)",
            identity=f"{scalar_type}(0.0)",
            name="mean",
        )
    else:
        raise ValueError

    return _get_or_create_kernel(expr)(a, out, axis=axis, keepdims=keepdims)


def var(a: ndarray, axis=None, dtype=None, out=None, keepdims=False):
    if dtype is None:
        dtype = np.dtype(np.float32)
    else:
        dtype = np.dtype(dtype)
    if dtype != np.dtype(np.float32):
        raise NotImplementedError
    expr = ReductionExpr(
        in_params="T x",
        out_params=f"float y",
        map_expr=f"vec2(x, x * x)",
        reduce_expr="a + b",
        post_map_expr=f"y = a.y / float(_in_ind_size / _out_ind_size) - (a.x * a.x) / (float(_in_ind_size / _out_ind_size) * float(_in_ind_size / _out_ind_size))",
        identity=f"vec2(0, 0)",
        name="var",
        reduce_type="vec2",
    )

    return _get_or_create_kernel(expr)(a, out, axis=axis, keepdims=keepdims)


def max(a: ndarray, axis=None, dtype=None, out=None, keepdims=False):
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = np.dtype(dtype)
    scalar_type = native_scalar_type_for_dtype[dtype]
    if scalar_type == "bool":
        expr = ReductionExpr(
            in_params="T x",
            out_params=f"{scalar_type} y",
            map_expr=f"{scalar_type}(x)",
            reduce_expr="a || b",
            post_map_expr="y = a",
            identity=f"false",
            name="max",
        )
    else:
        # FLT_MAX=3.402823e+38
        # GLSL spec: There is no limit on the number of digits in any digit-sequence. If the value of the floating point number
        # is too large (small) to be stored as a single precision value, it is converted to positive (negative) infinity.
        negative_inf = {"uint": "0", "int": "-2147483648", "float": "-1.0e+39"}[
            scalar_type
        ]
        expr = ReductionExpr(
            in_params="T x",
            out_params=f"{scalar_type} y",
            map_expr=f"{scalar_type}(x)",
            reduce_expr="max(a, b)",
            post_map_expr="y = a",
            identity=negative_inf,
            name="max",
        )

    return _get_or_create_kernel(expr)(a, out, axis=axis, keepdims=keepdims)


def min(a: ndarray, axis=None, dtype=None, out=None, keepdims=False):
    if dtype is None:
        dtype = a.dtype
    else:
        dtype = np.dtype(dtype)
    scalar_type = native_scalar_type_for_dtype[dtype]
    if scalar_type == "bool":
        expr = ReductionExpr(
            in_params="T x",
            out_params=f"{scalar_type} y",
            map_expr=f"{scalar_type}(x)",
            reduce_expr="a && b",
            post_map_expr="y = a",
            identity=f"true",
            name="min",
        )
    else:
        positive_inf = {"uint": "255", "int": "2147483647", "float": "1.0e+39"}[
            scalar_type
        ]
        expr = ReductionExpr(
            in_params="T x",
            out_params=f"{scalar_type} y",
            map_expr=f"{scalar_type}(x)",
            reduce_expr="min(a, b)",
            post_map_expr="y = a",
            identity=positive_inf,
            name="min",
        )

    return _get_or_create_kernel(expr)(a, out, axis=axis, keepdims=keepdims)
