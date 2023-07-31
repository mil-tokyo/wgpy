from typing import List
import numpy as np


class WebGPUArrayTextureShape:
    byte_length: int
    wgsl_dtype: str # data type in WGSL. f32, i32, u32 (bool is not used for buffer type. u32 is used for bool. f16 may be supported in the future.)
    itemsize: int
    # TODO: make class immutable

    def __init__(self, byte_length: int, wgsl_dtype: str) -> None:
        self.byte_length = byte_length
        self.wgsl_dtype = wgsl_dtype
        try:
            self.itemsize = {'f32': 4, 'i32': 4, 'u32': 4}[wgsl_dtype]
        except KeyError:
            raise ValueError(f'wgsl_dtype {wgsl_dtype} is not supported.')
    
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

_shape_queue = [] # type: List[WebGPUArrayTextureShape]

def enqueue_default_texture_shape(*texture_shapes: WebGPUArrayTextureShape):
    """
    Fixes the WebGPUArrayTextureShape obtained the next time get_default_texture_shape is called to a specific value.
    """
    _shape_queue.extend(texture_shapes)

def get_default_texture_shape(size: int, dtype: np.dtype) -> WebGPUArrayTextureShape:
    if len(_shape_queue) > 0:
        return _shape_queue.pop(0)
    
    if dtype == np.float32:
        wgsl_dtype = 'f32'
    elif dtype == np.int32:
        wgsl_dtype = 'i32'
    elif dtype == np.uint8:
        wgsl_dtype = 'u32'
    elif dtype == np.bool_:
        wgsl_dtype = 'u32'
    
    byte_length = max(size * 4, 4) # 0 byte is not allowed
    
    # currently, regardless of logical type, 4 bytes / element are allocated.
    return WebGPUArrayTextureShape(byte_length=byte_length, wgsl_dtype=wgsl_dtype)
