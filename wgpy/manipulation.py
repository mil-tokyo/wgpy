from typing import List, Optional, Tuple, Union
import numpy as np
from wgpy_backends.runtime.ndarray import ndarray
from wgpy.construct import asarray, asnumpy
from wgpy.common.shape_util import axis_to_tuple, calculate_c_contiguous_strides


def reshape(a: ndarray, newshape: Tuple[int, ...]) -> ndarray:
    # newshape can include -1
    minus_axis = None
    nonnegative_prod = 1
    for d in range(len(newshape)):
        if newshape[d] < 0:
            assert minus_axis is None
            minus_axis = d
        else:
            nonnegative_prod *= newshape[d]
    if minus_axis is not None:
        assert nonnegative_prod > 0
        assert a.size % nonnegative_prod == 0
        minus_val = a.size // nonnegative_prod
        ss = list(newshape)
        ss[minus_axis] = minus_val
        newshape = tuple(ss)
    else:
        assert nonnegative_prod == a.size

    if a.flags.c_contiguous:
        # make view
        return a.get_view(
            newshape,
            a.dtype,
            calculate_c_contiguous_strides(newshape, a.itemsize),
            a.offset,
        )
    else:
        # copy
        copy = a.copy()
        # TODO: specify shape and strides for copy itself, not make view
        return copy.get_view(
            newshape,
            copy.dtype,
            calculate_c_contiguous_strides(newshape, copy.itemsize),
            copy.offset,
        )


def transpose(a: ndarray, axes=None) -> ndarray:
    if axes is None or len(axes) == 0:
        axes = list(range(a.ndim)[::-1])
    assert len(axes) == a.ndim

    newshape = tuple(a.shape[axes[d]] for d in range(a.ndim))
    newstrides = tuple(a.strides[axes[d]] for d in range(a.ndim))
    return a.get_view(newshape, a.dtype, newstrides, a.offset)


def ravel(a: ndarray) -> ndarray:
    return reshape(a, (-1,))


def rollaxis(a: ndarray, axis: int, start: int = 0) -> ndarray:
    if start < 0:
        start = start + a.ndim
    if start < 0 or start > a.ndim:
        raise np.AxisError(axis=start)
    if axis < 0 or axis >= a.ndim:
        raise np.AxisError(axis=axis)
    newshape = list(a.shape)
    newstrides = list(a.strides)
    t_shape = newshape.pop(axis)
    t_strides = newstrides.pop(axis)
    if start > axis:
        start -= 1
    newshape.insert(start, t_shape)
    newstrides.insert(start, t_strides)

    return a.get_view(tuple(newshape), a.dtype, tuple(newstrides), a.offset)


def expand_dims(a: ndarray, axis: Union[int, Tuple[int]]) -> ndarray:
    axis = axis_to_tuple(axis)
    newshape = []
    newstrides = []
    orig_i = 0
    for d in range(len(axis) + a.ndim):
        if d in axis:
            newshape.append(1)
            if orig_i < a.ndim:
                newstrides.append(a.shape[orig_i] * a.strides[orig_i])
            else:
                newstrides.append(a.itemsize)
        else:
            newshape.append(a.shape[orig_i])
            newstrides.append(a.strides[orig_i])
            orig_i += 1
    return a.get_view(tuple(newshape), a.dtype, tuple(newstrides), a.offset)


def squeeze(a: ndarray, axis: Optional[Union[int, Tuple[int]]] = None) -> ndarray:
    if axis is not None:
        axis = axis_to_tuple(axis)
    newshape = []
    newstrides = []

    for d in range(a.ndim):
        remove = False
        if axis is None:
            if a.shape[d] == 1:
                remove = True
        else:
            if d in axis:
                if a.shape[d] != 1:
                    raise ValueError(
                        "cannot select an axis to squeeze out which has size not equal to one"
                    )
                remove = True
        if not remove:
            newshape.append(a.shape[d])
            newstrides.append(a.strides[d])
    return a.get_view(tuple(newshape), a.dtype, tuple(newstrides), a.offset)


def broadcast_to(array: ndarray, shape: Union[tuple, int], subok=False) -> ndarray:
    return array.broadcast_to(shape, subok)


def concatenate(
    arrays: List[ndarray], axis: int = 0, out=None, dtype=None, casting="same_kind"
) -> ndarray:
    # TODO implement on GPU
    if out is not None:
        raise ValueError("out is not supported")
    if dtype is not None:
        raise ValueError("dtype is not supported")
    if casting != "same_kind":
        raise ValueError("casting is not supported")
    np_arrays = [asnumpy(a) for a in arrays]
    return asarray(np.concatenate(np_arrays, axis=axis))


__all__ = [
    "broadcast_to",
    "expand_dims",
    "rollaxis",
    "reshape",
    "transpose",
    "ravel",
    "squeeze",
    "concatenate",
]
