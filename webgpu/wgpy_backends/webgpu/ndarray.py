from typing import Optional, Tuple, Union
import numpy as np
from wgpy_backends.webgpu.webgpu_data_type import WebGPULogicalDType, WebGPUStorageDType
from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape
from wgpy.common.ndarray_base import NDArrayBase, A
from wgpy_backends.webgpu.device import Device
from wgpy.common.indexing import calc_strides
from wgpy.common.shape_util import args_to_tuple_of_int, calculate_c_contiguous_strides
from wgpy_backends.webgpu.webgpu_buffer import WebGPUBuffer

class WebGPUArrayFlags:
    owndata: bool
    c_contiguous: bool
    c_contiguous_full: bool # c-contiguous and offset==0 and size==buffer.size
    f_contiguous: bool

    def __init__(self, owndata: bool, c_contiguous: bool, c_contiguous_full: bool, f_contiguous: bool) -> None:
        self.owndata = owndata
        self.c_contiguous = c_contiguous
        self.c_contiguous_full = c_contiguous_full
        self.f_contiguous = f_contiguous
    
    def __getitem__(self, key: str):
        if key == 'OWNDATA':
            return self.owndata
        elif key == 'C_CONTIGUOUS':
            return self.c_contiguous
        elif key == 'C_CONTIGUOUS_FULL':
            return self.c_contiguous_full
        elif key == 'F_CONTIGUOUS':
            return self.f_contiguous
        raise KeyError

class ndarray(NDArrayBase):
    base: Optional['ndarray']
    buffer: WebGPUBuffer
    flags: WebGPUArrayFlags

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, strides: Optional[Tuple[int, ...]]=None, offset: int=0, base: Optional['ndarray']=None, buffer: Optional[WebGPUBuffer]=None, owndata: Optional[bool]=None) -> None:
        super().__init__()
        assert isinstance(shape, tuple)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.ndim = len(shape)
        size = 1
        for i in range(self.ndim):
            size *= shape[i]
        self.size = size #int(np.prod(shape))
        self.itemsize = self.dtype.itemsize
        self.offset = offset
        if strides is None:
            strides = calculate_c_contiguous_strides(self.shape, self.itemsize)
        else:
            assert isinstance(strides, tuple)
        self.strides = strides
        self.nbytes = self.size * self.itemsize # not necessarily buffer size

        self.device = Device()
        self.base = base
        if owndata is None:
            # when owndata flag is not given, assume self owns the buffer if buffer is newly created
            owndata = buffer is None
        if buffer is None:
            buffer = WebGPUBuffer(self.size, self.dtype)
        self.buffer = buffer
        # TODO avoid circular referencing
        from wgpy_backends.webgpu.webgpu_array_func import WebGPUArrayFunc
        self.array_func = WebGPUArrayFunc.instance()
        c_contiguous = self._check_c_contiguous()
        self.flags = WebGPUArrayFlags(
            owndata=owndata,
            c_contiguous=c_contiguous,
            c_contiguous_full=c_contiguous and self.offset == 0 and self.size == self.buffer.size,
            f_contiguous=self._check_f_contiguous()
        )

    _astype_kernel = None
    def astype(self, dtype, *args, copy=True, **kwargs):
        raise NotImplementedError
        # if copy is False and dtype == self.dtype:
        #     return self
        # if ndarray._astype_kernel is None:
        #     from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
        #     ndarray._astype_kernel = ElementwiseKernel(
        #         in_params="T in0",
        #         out_params="V out0",
        #         operation="out0 = V(in0)",
        #         name="astype"
        #     )
        # from wgpy.construct import empty
        # out = empty(self.shape, dtype=dtype)
        # return ndarray._astype_kernel(self, out)
    
    def copy(self):
        return +self # __pos__
    
    def get_view(self: A, shape: Tuple[int, ...], dtype: np.dtype, strides: Tuple[int, ...], offset: int) -> A:
        return ndarray(shape=shape, dtype=dtype, strides=strides, offset=offset, buffer=self.buffer, owndata=False, base=self if self.base is None else self.base)

    def set(self, value: np.ndarray, stream=None):
        # ignore stream
        self.set_data(value)
    
    def set_data(self, data: np.ndarray):
        if self._is_full_view():
            back = np.empty((self.size, ), dtype=self.dtype)
            view = np.lib.stride_tricks.as_strided(back, self.shape, self.strides)
            view[...] = data
            self.buffer.set_data(back)
        else:
            # partial overwrite
            back = self.buffer.get_data().astype(self.dtype)
            view = np.lib.stride_tricks.as_strided(back[self.offset//data.dtype.itemsize:], self.shape, self.strides)
            view[...] = data
            self.buffer.set_data(back)

    def get_data(self) -> np.ndarray:
        # TODO Efficiency Improvement. Improve the mechanism to load the entire texture into the CPU, even if it is only one element view.
        data = self.buffer.get_data().astype(self.dtype, copy=False)
        # If strides is negative, element before data[self.offset//data.dtype.itemsize:][0] may be referred.
        view = np.lib.stride_tricks.as_strided(data[self.offset//data.dtype.itemsize:], self.shape, self.strides)
        return view.copy()

    def __getitem__(self: A, idxs) -> A:
        normalized_basic_idxs = _normalize_idxs_basic(idxs)
        if normalized_basic_idxs is not NonUnit:
            shape, strides, offset = calc_strides(idxs, self.shape, self.strides, self.offset)
            return self.get_view(shape, self.dtype, strides, offset)
        else:
            # advanced indexing
            normalized_advanced_idxs = _normalize_idxs_advanced(idxs)
            cpu_array = self.get()[normalized_advanced_idxs]
            gpu_array = ndarray(cpu_array.shape, cpu_array.dtype)
            gpu_array.set(cpu_array)
            return gpu_array

    def __setitem__(self, idxs, value) -> None:
        normalized_basic_idxs = _normalize_idxs_basic(idxs)
        if normalized_basic_idxs is not NonUnit:
            view = self[idxs]
            if isinstance(value, ndarray):
                value = value.get()
            view.set(value)
        else:
            # advanced indexing
            normalized_advanced_idxs = _normalize_idxs_advanced(idxs)
            cpu_array = self.get()
            if isinstance(value, ndarray):
                value = value.get()
            cpu_array[normalized_advanced_idxs] = value
            self.set(cpu_array)

class NonUnit_:
    pass

NonUnit = NonUnit_()

def _normalize_unit_basic(item):
    if item is None or item is Ellipsis or isinstance(item, int):
        return item
    if isinstance(item, slice):
        return slice(_normalize_unit_basic(item.start), _normalize_unit_basic(item.stop), _normalize_unit_basic(item.step))
    if isinstance(item, np.ndarray):
        if item.ndim == 0:
            return int(item)
        else:
            return NonUnit
    if isinstance(item, ndarray):
        if item.ndim == 0:
            return int(item)
        else:
            return NonUnit
    return NonUnit

def _normalize_idxs_basic(idxs):
    # basic indexing
    u = _normalize_unit_basic(idxs)
    if u is not NonUnit:
        return u
    if isinstance(idxs, tuple):
        us = []
        for idx in idxs:
            u = _normalize_unit_basic(idx)
            if u is NonUnit:
                break
            us.append(u)
        else:
            return tuple(us)

    return NonUnit

def _normalize_idxs_advanced(idxs):
    if idxs is None or idxs is Ellipsis or isinstance(idxs, int):
        return idxs
    if isinstance(idxs, slice):
        return slice(_normalize_unit_basic(idxs.start), _normalize_unit_basic(idxs.stop), _normalize_unit_basic(idxs.step))
    if isinstance(idxs, np.ndarray):
        if idxs.ndim == 0:
            return int(idxs)
        else:
            return idxs
    if isinstance(idxs, ndarray):
        if idxs.ndim == 0:
            return int(idxs)
        else:
            return idxs.get()
    if isinstance(idxs, tuple):
        return tuple(_normalize_idxs_advanced(i) for i in idxs)
    if isinstance(idxs, list):
        return [_normalize_idxs_advanced(i) for i in idxs]
    return idxs
