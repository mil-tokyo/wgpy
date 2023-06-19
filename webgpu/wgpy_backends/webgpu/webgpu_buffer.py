from collections import defaultdict
from typing import Optional
import numpy as np
from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape, get_default_texture_shape
from wgpy_backends.webgpu.platform import get_platform

performance_metrics = {
    'webgpu.buffer.create': 0,
    'webgpu.buffer.delete': 0,
    'webgpu.buffer.write_count': 0,
    'webgpu.buffer.write_size': 0,
    'webgpu.buffer.write_scalar_count': 0,
    'webgpu.buffer.read_count': 0,
    'webgpu.buffer.read_size': 0,
    'webgpu.buffer.read_scalar_count': 0,
    'webgpu.buffer.buffer_count': 0,
    'webgpu.buffer.buffer_count_max': 0,
    'webgpu.buffer.buffer_size': 0,
    'webgpu.buffer.buffer_size_max': 0,
}

class GPUBufferUsage:
    MAP_READ = 0x0001
    MAP_WRITE = 0x0002
    COPY_SRC = 0x0004
    COPY_DST = 0x0008
    INDEX = 0x0010
    VERTEX = 0x0020
    UNIFORM = 0x0040
    STORAGE = 0x0080
    INDIRECT = 0x0100
    QUERY_RESOLVE = 0x0200

_pool = defaultdict(list)

added_kernels = set()

def _pool_put(texture_shape: WebGPUArrayTextureShape, buffer_id: int):
    _pool[texture_shape].append(buffer_id)

def _pool_get(texture_shape: WebGPUArrayTextureShape) -> Optional[int]:
    if len(_pool[texture_shape]) > 0:
        return _pool[texture_shape].pop()
    return None

class WebGPUBuffer:
    buffer_id: int
    size: int # Logical number of elements (May differ from the number of elements in the physical buffer)
    dtype: np.dtype # logical type (may be different from physical representation in WebGPU)
    texture_shape: WebGPUArrayTextureShape

    _comm_buf: Optional[np.ndarray] = None
    next_id = 1

    def __init__(self, size: int, dtype: np.dtype, texture_shape: Optional[WebGPUArrayTextureShape]=None) -> None:
        self.size = size
        self.dtype = dtype
        self.texture_shape = texture_shape or get_default_texture_shape(size, dtype)
        pooled_buffer_id = _pool_get(self.texture_shape)
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGPUBuffer.next_id
            WebGPUBuffer.next_id += 1
            get_platform().createBuffer(self.buffer_id, self.texture_shape.byte_length, self.texture_shape.for_write_from_cpu, self.texture_shape.for_read_to_cpu)
            performance_metrics['webgpu.buffer.create'] += 1
            performance_metrics['webgpu.buffer.buffer_count'] += 1
            performance_metrics['webgpu.buffer.buffer_size'] += self.texture_shape.byte_length
            performance_metrics['webgpu.buffer.buffer_count_max'] = max(performance_metrics['webgpu.buffer.buffer_count_max'], performance_metrics['webgpu.buffer.buffer_count'])
            performance_metrics['webgpu.buffer.buffer_size_max'] = max(performance_metrics['webgpu.buffer.buffer_size_max'], performance_metrics['webgpu.buffer.buffer_size'])

    def __del__(self):
        # TODO: limit pooled size
        _pool_put(self.texture_shape, self.buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)
    
    def _get_comm_buf(self, byte_size: int) -> np.ndarray:
        if WebGPUBuffer._comm_buf is None or WebGPUBuffer._comm_buf.size < byte_size:
            WebGPUBuffer._comm_buf = np.empty((max(byte_size, 1024 * 1024),), dtype=np.uint8)
            get_platform().setCommBuf(WebGPUBuffer._comm_buf)
        return WebGPUBuffer._comm_buf
    
    def set_data(self, array: np.ndarray):
        if self.size == 0:
            return
        # TODO テンポラリバッファを作成してそこに書き込み、そこから本来のバッファにコピーする
        # cast array of type float to float32, int or bool to int32
        dtype = {np.dtype(np.bool_): np.dtype(np.int32), np.dtype(np.uint8): np.dtype(np.int32), np.dtype(np.int32): np.dtype(np.int32), np.dtype(np.float32): np.dtype(np.float32)}[self.dtype]
        buf = self._get_comm_buf(self.texture_shape.byte_length)
        packed = buf.view(dtype)
        packed[:array.size] = array.ravel()
        get_platform().setData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics['webgpu.buffer.write_count'] += 1
        # physical size
        performance_metrics['webgpu.buffer.write_size'] += self.texture_shape.byte_length
        # logical size
        if array.size <= 1:
            performance_metrics['webgpu.buffer.write_scalar_count'] += 1

    def get_data(self) -> np.ndarray:
        # TODO Sorting out dtype, whether it is a WebGPU internal representation or ndarray dtype.
        # TODO テンポラリバッファを作成してコピーし、そこから読み取る
        return self._get_data_internal(self.dtype)
        
    def _get_data_internal(self, original_dtype: np.dtype):
        if self.size == 0:
            return np.zeros((0,), dtype=original_dtype)
        performance_metrics['webgpu.buffer.read_count'] += 1
        dtype = {np.dtype(np.bool_): np.dtype(np.int32), np.dtype(np.uint8): np.dtype(np.int32), np.dtype(np.int32): np.dtype(np.int32), np.dtype(np.float32): np.dtype(np.float32)}[self.dtype]
        buf = self._get_comm_buf(self.texture_shape.byte_length)
        get_platform().getData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics['webgpu.buffer.read_size'] += self.texture_shape.byte_length
        if self.size <= 1:
            performance_metrics['webgpu.buffer.read_scalar_count'] += 1
        view = buf.view(dtype)[:self.size]

        return view.copy().astype(original_dtype, copy=False)
