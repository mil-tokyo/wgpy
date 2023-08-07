from typing import List
import numpy as np

from wgpy_backends.webgpu.webgpu_data_type import WebGPULogicalDType, WebGPUStorageDType

_dtype_to_logical_dtype = {
    np.dtype(np.bool_): 'bool',
    np.dtype(np.uint8): 'u32',
    np.dtype(np.uint32): 'u32',
    np.dtype(np.int32): 'i32',
    np.dtype(np.int64): 'i32', # may overflow
    np.dtype(np.float16): 'f32', # f16 may be used in the future (platform-dependent)
    np.dtype(np.float32): 'f32',
    np.dtype(np.float64): 'f32', # may overflow
}

_dtype_to_storage_dtype = {
    np.dtype(np.bool_): 'u32', # bool is not used for buffer type. u32 is used for bool.
    np.dtype(np.uint8): 'u32',
    np.dtype(np.uint32): 'u32',
    np.dtype(np.int32): 'i32',
    np.dtype(np.int64): 'i32', # may overflow
    np.dtype(np.float16): 'f32', # f16 may be used in the future (platform-dependent)
    np.dtype(np.float32): 'f32',
    np.dtype(np.float64): 'f32', # may overflow
}

_storage_dtype_to_itemsize = {
    'f32': 4,
    'i32': 4,
    'u32': 4,
}

class WebGPUArrayTextureShape:
    byte_length: int
    logical_dtype: WebGPULogicalDType # dtype in local vairable in webgpu
    storage_dtype: WebGPUStorageDType # dtype in storage in webgpu
    itemsize: int # byte size of each element in storage
    # TODO: make class immutable

    def __init__(self, byte_length: int, logical_dtype: WebGPULogicalDType, storage_dtype: WebGPUStorageDType) -> None:
        self.byte_length = byte_length
        self.logical_dtype = logical_dtype
        self.storage_dtype = storage_dtype
        try:
            self.itemsize = _storage_dtype_to_itemsize[storage_dtype]
        except KeyError:
            raise ValueError(f'storage_dtype {storage_dtype} is not supported.')

    def __eq__(self, other) -> bool:
        if not isinstance(other, WebGPUArrayTextureShape):
            return False
        return self._get_tuple() == other._get_tuple()
    
    def __hash__(self) -> int:
        return hash(self._get_tuple())
    
    def _get_tuple(self):
        return (self.byte_length, )

    def to_json(self):
        return {'byteLength': self.byte_length}
    
    @property
    def storage_dtype_numpy(self) -> np.dtype:
        return {'f32': np.dtype(np.float32), 'i32': np.dtype(np.int32), 'u32': np.dtype(np.uint32)}[self.storage_dtype]


_shape_queue = [] # type: List[WebGPUArrayTextureShape]

def enqueue_default_texture_shape(*texture_shapes: WebGPUArrayTextureShape):
    """
    Fixes the WebGPUArrayTextureShape obtained the next time get_default_texture_shape is called to a specific value.
    """
    _shape_queue.extend(texture_shapes)

def get_default_texture_shape(size: int, dtype: np.dtype) -> WebGPUArrayTextureShape:
    if len(_shape_queue) > 0:
        return _shape_queue.pop(0)
    
    logical_dtype = _dtype_to_logical_dtype[dtype]
    storage_dtype = _dtype_to_storage_dtype[dtype]
    
    byte_length = max(size * _storage_dtype_to_itemsize[storage_dtype], 4) # 0 byte is not allowed
    
    # currently, regardless of logical type, 4 bytes / element are allocated.
    return WebGPUArrayTextureShape(byte_length=byte_length, logical_dtype=logical_dtype, storage_dtype=storage_dtype)
