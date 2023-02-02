from typing import Optional, Tuple, Union
import numpy as np
from wgpy.common.ndarray_base import NDArrayBase
from wgpy_backends.webgl.device import Device
from wgpy.common.indexing import calc_strides
from wgpy.common.shape_util import args_to_tuple_of_int, calculate_c_contiguous_strides
from wgpy_backends.webgl.webgl_buffer import WebGLBuffer

pass_types = set(np.dtype(t) for t in [np.bool_, np.uint8, np.int32, np.float32])
type_maps = {
    np.dtype(np.float64): np.dtype(np.float32),
    np.dtype(np.int64): np.dtype(np.int32),
}

def map_dtype_to_webgl(dtype: np.dtype) -> np.dtype:
    if dtype in pass_types:
        return dtype
    new_dtype = type_maps.get(dtype)
    if new_dtype is None:
        raise NotImplementedError(f"dtype={dtype} is not implemented in webgl.")
    return new_dtype

class WebGLArrayFlags:
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
    flags: WebGLArrayFlags

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, strides: Optional[Tuple[int, ...]]=None, offset: int=0, base: Optional['ndarray']=None, buffer: Optional[WebGLBuffer]=None, owndata: Optional[bool]=None) -> None:
        super().__init__()
        assert isinstance(shape, tuple)
        self.shape = shape
        # TODO: Leave dtype as the requested type and provide a separate attribute for the physical type
        self.dtype = map_dtype_to_webgl(np.dtype(dtype))
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
            buffer = WebGLBuffer(self.size, self.dtype)
        self.buffer = buffer
        # TODO avoid circular referencing
        from wgpy_backends.webgl.webgl_array_func import WebGLArrayFunc
        self.array_func = WebGLArrayFunc.instance()
        c_contiguous = self._check_c_contiguous()
        self.flags = WebGLArrayFlags(
            owndata=owndata,
            c_contiguous=c_contiguous,
            c_contiguous_full=c_contiguous and self.offset == 0 and self.size == self.buffer.size,
            f_contiguous=self._check_f_contiguous()
        )

    @property
    def T(self):
        return self.transpose()

    _astype_kernel = None
    def astype(self, dtype, *args, copy=True, **kwargs):
        if copy is False and dtype == self.dtype:
            return self
        if ndarray._astype_kernel is None:
            from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel
            ndarray._astype_kernel = ElementwiseKernel(
                in_params="T in0",
                out_params="V out0",
                operation="out0 = V(in0)",
                name="astype"
            )
        from wgpy.construct import empty
        out = empty(self.shape, dtype=dtype)
        return ndarray._astype_kernel(self, out)
    
    def copy(self):
        return +self # __pos__
    
    def get_view(self, shape: Tuple[int, ...], dtype: np.dtype, strides: Tuple[int, ...], offset: int):
        return ndarray(shape=shape, dtype=dtype, strides=strides, offset=offset, buffer=self.buffer, owndata=False, base=self if self.base is None else self.base)

    def set(self, value: np.ndarray, stream=None):
        # ignore stream
        self.set_data(value)
    
    def get(self, stream=None, order='C', out=None) -> np.ndarray:
        # ignore stream
        assert order == 'C'
        assert out is None
        return self.get_data()

    def to_cpu(self) -> np.ndarray:
        return self.get()
    
    def _is_full_view(self) -> bool:
        if self.offset != 0:
            return False
        if self.size != self.buffer.size:
            return False
        return True

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

    def broadcast_to(self, shape: Union[tuple, int], subok=False) -> 'ndarray':
        if isinstance(shape, int):
            shape = (shape,)
        orig_shape = list(self.shape)
        assert len(orig_shape) <= len(shape)
        new_strides = list(self.strides)
        while len(orig_shape) < len(shape):
            # append 1 to head
            orig_shape.insert(0, 1)
            new_strides.insert(0, 0)
        for d in range(len(shape)):
            if orig_shape[d] == 1 and shape[d] != 1:
                # broadcasted axis
                new_strides[d] = 0
        return self.get_view(shape, self.dtype, tuple(new_strides), self.offset)

    def dot(self, other, out=None):
        # TODO: avoid circular reference
        from wgpy.binary import dot
        return dot(self, other, out=out)

    def max(self, *args, **kwargs):
        from wgpy.reduction import max
        return max(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        from wgpy.construct import asarray, asnumpy
        return asarray(asnumpy(self).argmax(*args, **kwargs))

    def reshape(self, *shape):
        # array.reshape(2,3) and array.reshape((2,3)) are ok
        from wgpy.manipulation import reshape
        return reshape(self, args_to_tuple_of_int(shape))

    def sum(self, *args, **kwargs):
        from wgpy.reduction import sum
        return sum(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        from wgpy.reduction import mean
        return mean(self, *args, **kwargs)

    def var(self, *args, **kwargs):
        from wgpy.reduction import var
        return var(self, *args, **kwargs)

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]]=None):
        from wgpy.manipulation import squeeze
        return squeeze(self, axis)

    def transpose(self, *axes):
        from wgpy.manipulation import transpose
        return transpose(self, args_to_tuple_of_int(axes))

    def ravel(self):
        from wgpy.manipulation import ravel
        return ravel(self)
    
    def reduced_view(self):
        # not exist in numpy, but exist in cupy
        # https://github.com/cupy/cupy/issues/2114
        if self.ndim == 0:
            # shape == () => ()
            shape = ()
            strides = ()
        elif self.ndim == 1:
            shape = self.shape
            strides = self.strides
        elif self.size == 0:
            shape = (0,)
            strides = (self.itemsize,)
        elif self.size == 1:
            # when ndim==1, shape=(1,)
            shape = ()
            strides = ()
        else:
            # generic case
            new_shape = list(self.shape)
            new_strides = list(self.strides)
            while True:
                changed = False
                for d in range(len(new_shape)-1):
                    if new_strides[d] == new_strides[d+1] * new_shape[d+1]:
                        # dim d and d+1 can be combined
                        new_shape[d] = new_shape[d] * new_shape[d+1]
                        new_shape.pop(d+1)
                        new_strides.pop(d)
                        changed = True
                        break
                if not changed:
                    break
            shape = tuple(new_shape)
            strides = tuple(new_strides)
        return self.get_view(shape, dtype=self.dtype, strides=strides, offset=self.offset)

    def __getitem__(self, idxs) -> 'ndarray':
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

    def __setitem__(self, idxs, value):
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
