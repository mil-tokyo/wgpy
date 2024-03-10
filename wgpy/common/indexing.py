import math
from typing import Tuple


def ellipsis_to_slice(idxs, t_ndim: int) -> list:
    slice_count = 0
    has_ellipsis = False
    for i, idx in enumerate(idxs):
        if idx is Ellipsis:
            if has_ellipsis:
                raise ValueError("Multiple Ellipsis (...) found.")
            has_ellipsis = True
        elif isinstance(idx, slice) or isinstance(idx, int):
            slice_count += 1
        elif idx is None:
            pass
        else:
            raise ValueError(f"Index {idx} is not supported.")
    ndim_to_fill = t_ndim - slice_count
    if ndim_to_fill < 0:
        raise ValueError("The number of index exceeds array dimensions.")

    if has_ellipsis:
        idxs_wo_ellipsis = []
        for idx in idxs:
            if idx is Ellipsis:
                for _ in range(ndim_to_fill):
                    idxs_wo_ellipsis.append(slice(None))
            else:
                idxs_wo_ellipsis.append(idx)
    else:
        idxs_wo_ellipsis = idxs.copy()
        for _ in range(ndim_to_fill):
            idxs_wo_ellipsis.append(slice(None))
    return idxs_wo_ellipsis


def calc_strides(
    idxs, t_shape: Tuple[int, ...], t_strides: Tuple[int, ...], t_offset: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    if isinstance(idxs, (int, slice)) or idxs is None or idxs is Ellipsis:
        idxs = [idxs]
    elif isinstance(idxs, tuple):
        idxs = list(idxs)
    else:
        raise ValueError("array index must be int, slice, None, Ellipsis or tuple")
    idxs_wo_ellipsis = ellipsis_to_slice(idxs, len(t_shape))
    idxs_wo_new_axis = []
    new_axis_dims = []
    squeeze_dims = []
    for i, idx in enumerate(idxs_wo_ellipsis):
        if idx is None:
            new_axis_dims.append(i)
        else:
            idxs_wo_new_axis.append(idx)
            if isinstance(idx, int):
                squeeze_dims.append(i)

    # normalize slice
    assert len(idxs_wo_new_axis) == len(t_shape)
    v_shape = []
    step_strides = []
    step_offset_num = t_offset
    for i, (t_len, idx) in enumerate(zip(t_shape, idxs_wo_new_axis)):
        if isinstance(idx, int):
            start = idx if idx >= 0 else idx + t_len
            if start < 0 or start >= t_len:
                raise ValueError(
                    f"index {idx} for axis {i} is out of range for shape {t_shape}"
                )
            stop = start + 1
            step = 1
            v_len = 1
        else:
            step = idx.step or 1
            if step > 0:
                if idx.start is None:
                    start = 0
                else:
                    start = idx.start
                    if start < 0:
                        start += t_len
                    if start < 0:
                        start = 0
                    if start > t_len:
                        start = t_len
                if idx.stop is None:
                    stop = t_len
                else:
                    stop = idx.stop
                    if stop < 0:
                        stop += t_len
                    if stop < 0:
                        stop = 0
                    if stop > t_len:
                        stop = t_len
                if stop > start:
                    v_len = int(math.ceil((stop - start) / step))
                else:
                    v_len = 0
            else:  # step <= 0
                if idx.start is None:
                    start = t_len - 1
                else:
                    start = idx.start
                    if start < 0:
                        start += t_len
                    if start < 0:
                        start = 0
                    if start >= t_len:
                        start = t_len - 1
                if idx.stop is None:
                    stop = -1
                else:
                    stop = idx.stop
                    if stop < 0:
                        stop += t_len
                    if stop < 0:
                        stop = 0
                    if stop > t_len:
                        stop = t_len
                if stop < start:
                    v_len = int(math.ceil((stop - start) / step))
                else:
                    v_len = 0
        step_stride = t_strides[i] * step
        step_offset = t_strides[i] * start
        v_shape.append(v_len)
        step_strides.append(step_stride)
        step_offset_num += step_offset

    for axis in new_axis_dims:
        v_shape.insert(axis, 1)
        step_strides.insert(axis, 0)

    for axis in reversed(squeeze_dims):
        v_shape.pop(axis)
        step_strides.pop(axis)

    return tuple(v_shape), tuple(step_strides), step_offset_num
