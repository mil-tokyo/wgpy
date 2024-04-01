from collections import defaultdict
from typing import Optional
import numpy as np
from wgpy_backends.webgl.texture import (
    WebGL2RenderingContext,
    WebGLArrayTextureShape,
    get_default_texture_shape,
)
from wgpy_backends.webgl.shader_util import (
    header,
    native_pixel_type_for_internal_format,
)
from wgpy_backends.webgl.platform import get_platform
import wgpy_backends.webgl.webgl_config as webgl_config

performance_metrics = {
    "webgl.buffer.create": 0,
    "webgl.buffer.delete": 0,
    "webgl.buffer.write_count": 0,
    "webgl.buffer.write_size": 0,
    "webgl.buffer.write_scalar_count": 0,
    "webgl.buffer.read_count": 0,
    "webgl.buffer.read_size": 0,
    "webgl.buffer.read_scalar_count": 0,
    "webgl.buffer.buffer_count": 0,
    "webgl.buffer.buffer_count_max": 0,
    "webgl.buffer.buffer_size": 0,
    "webgl.buffer.buffer_size_max": 0,
}

texture_type_to_element_itemsize = {
    WebGL2RenderingContext.FLOAT: 4,
    WebGL2RenderingContext.HALF_FLOAT: 2,
    WebGL2RenderingContext.INT: 4,
    WebGL2RenderingContext.UNSIGNED_BYTE: 1,
}


def get_dtype_js_ctor_type(dtype):
    dtype = np.dtype(dtype)
    return {
        np.dtype(np.float32): "Float32Array",
        np.dtype(np.int32): "Int32Array",
        np.dtype(np.uint16): "Uint16Array",
        np.dtype(np.uint8): "Uint8Array",
    }[dtype]


_pool = defaultdict(list)

added_kernels = set()


def _pool_put(texture_shape: WebGLArrayTextureShape, buffer_id: int):
    _pool[texture_shape].append(buffer_id)


def _pool_get(texture_shape: WebGLArrayTextureShape) -> Optional[int]:
    if len(_pool[texture_shape]) > 0:
        return _pool[texture_shape].pop()
    return None


class WebGLBuffer:
    buffer_id: int
    size: int  # Logical number of elements (May differ from the number of elements in the texture; for RGBA textures, 1 pixel corresponds to 4 elements)
    dtype: (
        np.dtype
    )  # logical type (may be different from physical representation in WebGL)
    texture_shape: WebGLArrayTextureShape

    _comm_buf: Optional[np.ndarray] = None
    next_id = 1

    def __init__(
        self,
        size: int,
        dtype: np.dtype,
        texture_shape: Optional[WebGLArrayTextureShape] = None,
    ) -> None:
        self.size = size
        self.dtype = dtype
        self.texture_shape = texture_shape or get_default_texture_shape(size, dtype)
        pooled_buffer_id = _pool_get(self.texture_shape)
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGLBuffer.next_id
            WebGLBuffer.next_id += 1
            get_platform().createBuffer(self.buffer_id, self.texture_shape.to_json())
            performance_metrics["webgl.buffer.create"] += 1
            performance_metrics["webgl.buffer.buffer_count"] += 1
            performance_metrics["webgl.buffer.buffer_size"] += (
                self.texture_shape.element_count
                * texture_type_to_element_itemsize[self.texture_shape.type]
            )
            performance_metrics["webgl.buffer.buffer_count_max"] = max(
                performance_metrics["webgl.buffer.buffer_count_max"],
                performance_metrics["webgl.buffer.buffer_count"],
            )
            performance_metrics["webgl.buffer.buffer_size_max"] = max(
                performance_metrics["webgl.buffer.buffer_size_max"],
                performance_metrics["webgl.buffer.buffer_size"],
            )

    def __del__(self):
        # TODO: limit pooled size
        _pool_put(self.texture_shape, self.buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)

    def _get_comm_buf(self, byte_size: int) -> np.ndarray:
        if WebGLBuffer._comm_buf is None or WebGLBuffer._comm_buf.size < byte_size:
            WebGLBuffer._comm_buf = np.empty(
                (max(byte_size, 1024 * 1024),), dtype=np.uint8
            )
            get_platform().setCommBuf(WebGLBuffer._comm_buf)
        return WebGLBuffer._comm_buf

    def set_data(self, array: np.ndarray):
        if self.texture_shape.type == WebGL2RenderingContext.HALF_FLOAT:
            array_f16 = array.astype(np.float16).ravel()
            buf = self._get_comm_buf(
                np.dtype(np.uint16).itemsize * self.texture_shape.element_count
            )
            packed = buf.view(np.float16)
            packed[: array.size] = array_f16
            size = self.texture_shape.element_count
            dtype = np.uint16
        else:
            dtype = {
                WebGL2RenderingContext.FLOAT: np.float32,
                WebGL2RenderingContext.INT: np.int32,
                WebGL2RenderingContext.UNSIGNED_BYTE: np.uint8,
            }[self.texture_shape.type]
            buf = self._get_comm_buf(
                np.dtype(dtype).itemsize * self.texture_shape.element_count
            )
            packed = buf.view(dtype)
            packed[: array.size] = array.ravel()
            size = self.texture_shape.element_count
        get_platform().setData(self.buffer_id, get_dtype_js_ctor_type(dtype), size)
        performance_metrics["webgl.buffer.write_count"] += 1
        # physical size
        performance_metrics["webgl.buffer.write_size"] += (
            size * np.dtype(dtype).itemsize
        )
        # logical size
        if array.size <= 1:
            performance_metrics["webgl.buffer.write_scalar_count"] += 1

    def get_data(self) -> np.ndarray:
        # TODO Sorting out dtype, whether it is a WebGL internal representation or ndarray dtype.
        copied = self._copy_to_rgba_if_needed()
        if copied is not None:
            return copied._get_data_internal(
                self.texture_shape.elements_per_pixel == 1, self.dtype
            )
        else:
            return self._get_data_internal(False, self.dtype)

    def _get_data_internal(self, extract_r_from_rgba: bool, original_dtype: np.dtype):
        performance_metrics["webgl.buffer.read_count"] += 1
        if self.texture_shape.type == WebGL2RenderingContext.HALF_FLOAT:
            buf = self._get_comm_buf(
                np.dtype(np.uint16).itemsize * self.texture_shape.element_count
            )
            get_platform().getData(
                self.buffer_id,
                get_dtype_js_ctor_type(np.uint16),
                self.texture_shape.element_count,
            )
            performance_metrics["webgl.buffer.read_size"] += (
                self.texture_shape.element_count * np.dtype(np.uint16).itemsize
            )
            if self.size <= 1:
                performance_metrics["webgl.buffer.read_scalar_count"] += 1
            view = buf.view(np.float16)[: self.size]
            if extract_r_from_rgba:
                view = view[::4]

            return view.astype(np.float32).astype(original_dtype, copy=False)
        else:
            dtype = {
                WebGL2RenderingContext.FLOAT: np.float32,
                WebGL2RenderingContext.INT: np.int32,
                WebGL2RenderingContext.UNSIGNED_BYTE: np.uint8,
            }[self.texture_shape.type]
            buf = self._get_comm_buf(
                np.dtype(dtype).itemsize * self.texture_shape.element_count
            )
            get_platform().getData(
                self.buffer_id,
                get_dtype_js_ctor_type(dtype),
                self.texture_shape.element_count,
            )
            performance_metrics["webgl.buffer.read_size"] += (
                self.texture_shape.element_count * np.dtype(np.uint16).itemsize
            )
            if self.size <= 1:
                performance_metrics["webgl.buffer.read_scalar_count"] += 1
            view = buf.view(dtype)[: self.size]
            if extract_r_from_rgba:
                view = view[::4]

            return view.copy().astype(original_dtype, copy=False)

    def _copy_to_rgba_if_needed(self) -> Optional["WebGLBuffer"]:
        is_32bit = self.texture_shape.type in [
            WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.INT,
        ]
        is_rch = self.texture_shape.elements_per_pixel == 1
        ok = True
        if not webgl_config.can_read_r_texture() and is_rch:
            ok = False
        if not webgl_config.can_read_non_32bit_texture() and not is_32bit:
            ok = False
        if ok:
            # no copy needed
            return None

        ss = self.texture_shape  # source shape
        t_internal_format = {
            WebGL2RenderingContext.R16F: WebGL2RenderingContext.RGBA32F,  # some env does not support reading RGBA16F
            WebGL2RenderingContext.R32F: WebGL2RenderingContext.RGBA32F,
            WebGL2RenderingContext.R32I: WebGL2RenderingContext.RGBA32I,
            WebGL2RenderingContext.R8UI: WebGL2RenderingContext.RGBA32I,  # some env does not support reading 8UI
            WebGL2RenderingContext.RGBA16F: WebGL2RenderingContext.RGBA32F,  # some env does not support reading RGBA16F
            WebGL2RenderingContext.RGBA32F: WebGL2RenderingContext.RGBA32F,
            WebGL2RenderingContext.RGBA32I: WebGL2RenderingContext.RGBA32I,
            WebGL2RenderingContext.RGBA8UI: WebGL2RenderingContext.RGBA32I,  # some env does not support reading 8UI
        }[ss.internal_format]
        t_format = {
            WebGL2RenderingContext.RED: WebGL2RenderingContext.RGBA,
            WebGL2RenderingContext.RED_INTEGER: WebGL2RenderingContext.RGBA_INTEGER,
            WebGL2RenderingContext.RGBA: WebGL2RenderingContext.RGBA,
            WebGL2RenderingContext.RGBA_INTEGER: WebGL2RenderingContext.RGBA_INTEGER,
        }[ss.format]
        t_type = {
            WebGL2RenderingContext.FLOAT: WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.HALF_FLOAT: WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.INT: WebGL2RenderingContext.INT,
            WebGL2RenderingContext.UNSIGNED_BYTE: WebGL2RenderingContext.INT,  # 8UI -> 32I
        }[ss.type]
        target_shape = WebGLArrayTextureShape(
            height=ss.height,
            width=ss.width,
            depth=ss.depth,
            dim=ss.dim,
            internal_format=t_internal_format,
            format=t_format,
            type=t_type,
        )
        target_dtype = {
            np.dtype(np.float32): np.dtype(np.float32),
            np.dtype(np.int32): np.dtype(np.int32),
            np.dtype(np.uint8): np.dtype(np.int32),
            np.dtype(np.bool_): np.dtype(np.int32),
        }[self.dtype]
        # always get vec4 even it only contains R channel
        native_pixel_type_src = {
            WebGL2RenderingContext.R32F: "vec4",
            WebGL2RenderingContext.R16F: "vec4",
            WebGL2RenderingContext.R32I: "ivec4",
            WebGL2RenderingContext.R8UI: "uvec4",
            WebGL2RenderingContext.RGBA32F: "vec4",
            WebGL2RenderingContext.RGBA16F: "vec4",
            WebGL2RenderingContext.RGBA32I: "ivec4",
            WebGL2RenderingContext.RGBA8UI: "uvec4",
        }[ss.internal_format]
        native_pixel_type_dst = native_pixel_type_for_internal_format[
            target_shape.internal_format
        ]

        sampler_type = {
            WebGL2RenderingContext.FLOAT: "sampler2D",
            WebGL2RenderingContext.HALF_FLOAT: "sampler2D",
            WebGL2RenderingContext.INT: "isampler2D",
            WebGL2RenderingContext.UNSIGNED_BYTE: "usampler2D",
        }[ss.type]
        if ss.dim == "2DArray":
            sampler_type += "Array"

        new_size = (
            self.size if self.texture_shape.elements_per_pixel == 4 else self.size * 4
        )
        target_buffer = WebGLBuffer(
            size=new_size, dtype=target_dtype, texture_shape=target_shape
        )

        kernel_source = None
        if target_shape.dim == "2D":
            kernel_name = (
                f"copy_r_to_rgba_2d_{native_pixel_type_src}_{native_pixel_type_dst}"
            )
            if kernel_name not in added_kernels:
                kernel_source = f"""{header}
uniform {sampler_type} _src_r;
out {native_pixel_type_dst} _out_color;
void main() {{
{native_pixel_type_src} v = texelFetch(_src_r, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0);
_out_color = {native_pixel_type_dst}(v);
}}
"""
        else:
            kernel_name = f"copy_r_to_rgba_2darray_{native_pixel_type_src}_{native_pixel_type_dst}"
            if kernel_name not in added_kernels:
                kernel_source = f"""{header}
uniform {sampler_type} _src_r;
out {native_pixel_type_dst} _out_color;
uniform int _draw_depth;
void main() {{
{native_pixel_type_src} v = texelFetch(_src_r, ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), _draw_depth), 0);
_out_color = {native_pixel_type_dst}(v);
}}
"""
        if kernel_source is not None:
            get_platform().addKernel(kernel_name, {"source": kernel_source})
            added_kernels.add(kernel_name)

        get_platform().runKernel(
            {
                "name": kernel_name,
                "inputs": [{"name": "_src_r", "id": self.buffer_id}],
                "output": target_buffer.buffer_id,
                "uniforms": [],
            }
        )
        return target_buffer
