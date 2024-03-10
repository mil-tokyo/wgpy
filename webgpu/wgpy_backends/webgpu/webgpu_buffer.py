from collections import defaultdict
from typing import List, Optional
import numpy as np
from wgpy_backends.webgpu.webgpu_data_type import WebGPULogicalDType, WebGPUStorageDType
from wgpy_backends.webgpu.texture import (
    WebGPUArrayTextureShape,
    get_default_texture_shape,
)
from wgpy_backends.webgpu.platform import get_platform


performance_metrics = {
    "webgpu.buffer.create": 0,
    "webgpu.buffer.delete": 0,
    "webgpu.buffer.write_count": 0,
    "webgpu.buffer.write_size": 0,
    "webgpu.buffer.write_scalar_count": 0,
    "webgpu.buffer.read_count": 0,
    "webgpu.buffer.read_size": 0,
    "webgpu.buffer.read_scalar_count": 0,
    "webgpu.buffer.buffer_count": 0,
    "webgpu.buffer.buffer_count_max": 0,
    "webgpu.buffer.buffer_size": 0,
    "webgpu.buffer.buffer_size_max": 0,
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


def _get_comm_buf(byte_size: int) -> np.ndarray:
    if WebGPUBuffer._comm_buf is None or WebGPUBuffer._comm_buf.size < byte_size:
        WebGPUBuffer._comm_buf = np.empty(
            (max(byte_size, 1024 * 1024),), dtype=np.uint8
        )
        get_platform().setCommBuf(WebGPUBuffer._comm_buf)
    return WebGPUBuffer._comm_buf


class WebGPUBufferBase:
    buffer_id: int


class WebGPUBuffer(WebGPUBufferBase):
    size: int  # Logical number of elements (May differ from the number of elements in the physical buffer)
    dtype: (
        np.dtype
    )  # ndarray logical type (may be different from physical representation in WebGPU)
    texture_shape: WebGPUArrayTextureShape

    _comm_buf: Optional[np.ndarray] = None
    next_id = 1

    def __init__(
        self,
        size: int,
        dtype: np.dtype,
        texture_shape: Optional[WebGPUArrayTextureShape] = None,
    ) -> None:
        self.size = size
        self.dtype = dtype
        self.texture_shape = texture_shape or get_default_texture_shape(size, dtype)
        pooled_buffer_id = _pool_get(self.texture_shape)
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGPUBuffer.next_id
            WebGPUBuffer.next_id += 1
            get_platform().createBuffer(self.buffer_id, self.texture_shape.byte_length)
            performance_metrics["webgpu.buffer.create"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] += 1
            performance_metrics[
                "webgpu.buffer.buffer_size"
            ] += self.texture_shape.byte_length
            performance_metrics["webgpu.buffer.buffer_count_max"] = max(
                performance_metrics["webgpu.buffer.buffer_count_max"],
                performance_metrics["webgpu.buffer.buffer_count"],
            )
            performance_metrics["webgpu.buffer.buffer_size_max"] = max(
                performance_metrics["webgpu.buffer.buffer_size_max"],
                performance_metrics["webgpu.buffer.buffer_size"],
            )

    def __del__(self):
        # TODO: limit pooled size
        _pool_put(self.texture_shape, self.buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)

    def set_data(self, array: np.ndarray):
        if self.size == 0:
            return
        buf = _get_comm_buf(self.texture_shape.byte_length)
        packed = buf.view(self.texture_shape.storage_dtype_numpy)
        packed[: array.size] = array.ravel()
        get_platform().setData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics["webgpu.buffer.write_count"] += 1
        # physical size
        performance_metrics[
            "webgpu.buffer.write_size"
        ] += self.texture_shape.byte_length
        # logical size
        if array.size <= 1:
            performance_metrics["webgpu.buffer.write_scalar_count"] += 1

    def get_data(self) -> np.ndarray:
        return self._get_data_internal(self.dtype)

    def _get_data_internal(self, original_dtype: np.dtype):
        if self.size == 0:
            return np.zeros((0,), dtype=original_dtype)
        performance_metrics["webgpu.buffer.read_count"] += 1
        buf = _get_comm_buf(self.texture_shape.byte_length)
        get_platform().getData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics["webgpu.buffer.read_size"] += self.texture_shape.byte_length
        if self.size <= 1:
            performance_metrics["webgpu.buffer.read_scalar_count"] += 1
        view = buf.view(self.texture_shape.storage_dtype_numpy)[: self.size]

        return view.copy().astype(original_dtype, copy=False)


_meta_pool = defaultdict(list)


class WebGPUMetaBuffer(WebGPUBufferBase):
    _data: bytes

    def __init__(self, data: bytes, pooled_buffer_id: Optional[int]) -> None:
        super().__init__()

        self._data = data
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGPUBuffer.next_id
            WebGPUBuffer.next_id += 1

            buf = _get_comm_buf(len(data))
            packed = buf.view(np.uint8)
            packed[: len(data)] = np.frombuffer(data, dtype=np.uint8)
            get_platform().createMetaBuffer(self.buffer_id, len(data))

            performance_metrics["webgpu.buffer.create"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] += 1
            performance_metrics["webgpu.buffer.buffer_size"] += len(data)
            performance_metrics["webgpu.buffer.buffer_count_max"] = max(
                performance_metrics["webgpu.buffer.buffer_count_max"],
                performance_metrics["webgpu.buffer.buffer_count"],
            )
            performance_metrics["webgpu.buffer.buffer_size_max"] = max(
                performance_metrics["webgpu.buffer.buffer_size_max"],
                performance_metrics["webgpu.buffer.buffer_size"],
            )

    @property
    def data(self):
        return self._data

    def __del__(self):
        _meta_pool[self._data].append(self.buffer_id)


class WebGPUMetaBufferItem:
    name: str
    native_type: str
    numpy_dtype_str: str

    def __init__(
        self, name: str, native_type: str, numpy_dtype_str: Optional[str] = None
    ) -> None:
        self.name = name
        self.native_type = native_type
        self.numpy_dtype_str = (
            numpy_dtype_str or {"f32": "f4", "i32": "i4", "u32": "u4"}[native_type]
        )

    def __repr__(self) -> str:
        return f"WebGPUMetaBufferItem('{self.name}', '{self.native_type}', '{self.numpy_dtype_str}')"


def create_meta_buffer(data: bytes) -> WebGPUMetaBuffer:
    pooled = _meta_pool[data]
    pooled_buffer_id = None
    if len(pooled) > 0:
        pooled_buffer_id = pooled.pop()
    new_buf = WebGPUMetaBuffer(data, pooled_buffer_id=pooled_buffer_id)
    return new_buf


def create_meta_buffer_from_structure(data_tuple: tuple, dtype) -> WebGPUMetaBuffer:
    """
    example: data_tuple = (2, 1.5), dtype = "i4,f4"
    """
    structured_array = np.array([data_tuple], dtype=dtype)
    data = structured_array.tobytes()
    return create_meta_buffer(data)


def create_meta_buffer_from_dict(
    data_dict: dict, item_definitions: List[WebGPUMetaBufferItem]
) -> WebGPUMetaBuffer:
    """
    example: data_dict = {"a": 2, "b": 1.5}, item_definitions = [WebGPUMetaBufferItem("a", "i32"), WebGPUMetaBufferItem("b", "f32")]
    """
    dtype = np.dtype([(item.name, item.numpy_dtype_str) for item in item_definitions])
    structured_array = np.array(
        [tuple(data_dict[item.name] for item in item_definitions)], dtype=dtype
    )
    data = structured_array.tobytes()
    return create_meta_buffer(data)
