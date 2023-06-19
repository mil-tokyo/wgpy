from typing import List
import numpy as np


class WebGPUArrayTextureShape:
    byte_length: int
    # TODO: make class immutable

    def __init__(self, byte_length: int, for_write_from_cpu: bool, for_read_to_cpu: bool) -> None:
        self.byte_length = byte_length
        self.for_write_from_cpu = for_write_from_cpu
        self.for_read_to_cpu = for_read_to_cpu
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, WebGPUArrayTextureShape):
            return False
        return self._get_tuple() == other._get_tuple()
    
    def __hash__(self) -> int:
        return hash(self._get_tuple())
    
    def _get_tuple(self):
        return (self.byte_length, self.for_write_from_cpu, self.for_read_to_cpu)

    def to_json(self):
        return {'byteLength': self.byte_length, 'forWriteFromCPU': self.for_write_from_cpu, 'forReadToCPU': self.for_read_to_cpu}

_shape_queue = [] # type: List[WebGPUArrayTextureShape]

def enqueue_default_texture_shape(*texture_shapes: WebGPUArrayTextureShape):
    """
    Fixes the WebGPUArrayTextureShape obtained the next time get_default_texture_shape is called to a specific value.
    """
    _shape_queue.extend(texture_shapes)

def get_default_texture_shape(size: int, dtype: np.dtype, for_write_from_cpu: bool=False, for_read_to_cpu: bool=False) -> WebGPUArrayTextureShape:
    if len(_shape_queue) > 0:
        return _shape_queue.pop(0)
    
    byte_length = max(size * 4, 4) # 0 byte is not allowed
    
    # currently, regardless of logical type, 4 bytes / element are allocated.
    return WebGPUArrayTextureShape(byte_length=byte_length, for_write_from_cpu=for_write_from_cpu, for_read_to_cpu=for_read_to_cpu)
