import math
from typing import List, Optional
import numpy as np

from wgpy_backends.webgl.webgl_config import get_float_texture_bit, get_max_texture_size

class WebGL2RenderingContext:
    R32F = 33326
    RED = 6403
    FLOAT = 5126
    R16F = 33325
    HALF_FLOAT = 5131
    R32I = 33333
    RED_INTEGER = 36244
    INT = 5124
    R8UI = 33330
    UNSIGNED_BYTE = 5121
    RGBA32F = 34836
    RGBA = 6408
    RGBA16F = 34842
    HALF_FLOAT = 5131
    RGBA32I = 36226
    RGBA_INTEGER = 36249
    RGBA8UI = 36220

class WebGLArrayTextureShape:
    internal_format: int
    format: int
    type: int
    dim: str # '2D'|'2DArray'
    width: int
    height: int
    depth: int
    elements_per_pixel: int
    # TODO: make class immutable

    def __init__(self, height: int, width: int, depth: int = 1, dim: str = '2D', internal_format: Optional[int] = None, format: Optional[int] = None, type: Optional[int] = None) -> None:
        assert dim in ['2D', '2DArray']
        if dim == '2D':
            assert depth == 1
        self.height = height
        self.width = width
        self.depth = depth
        self.dim = dim
        if internal_format is None:
            internal_format = WebGL2RenderingContext.R16F if get_float_texture_bit() == 16 else WebGL2RenderingContext.R32F
        if format is None:
            format = WebGL2RenderingContext.RED
        if type is None:
            type = WebGL2RenderingContext.HALF_FLOAT if get_float_texture_bit() == 16 else WebGL2RenderingContext.FLOAT
        self.internal_format = internal_format
        self.format = format
        self.type = type
        self.elements_per_pixel = {WebGL2RenderingContext.RED: 1, WebGL2RenderingContext.RED_INTEGER: 1, WebGL2RenderingContext.RGBA: 4, WebGL2RenderingContext.RGBA_INTEGER: 4}[format]
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, WebGLArrayTextureShape):
            return False
        return self._get_tuple() == other._get_tuple()
    
    def __hash__(self) -> int:
        return hash(self._get_tuple())
    
    def _get_tuple(self):
        return (self.dim, self.width, self.height, self.depth, self.internal_format, self.format, self.type)

    def to_json(self):
        return {'dim': self.dim, 'width': self.width, 'height': self.height, 'depth': self.depth, 'internalFormat': self.internal_format, 'format': self.format, 'type': self.type}
    
    @property
    def element_count(self):
        """
        The number of elements in texture. RGBA is counted as 4.
        """
        return self.elements_per_pixel * self.depth * self.height * self.width

    @property
    def pixel_count(self):
        """
        The number of pixels, including depth
        """
        return self.depth * self.height * self.width

    @property
    def pixel_count_plane(self):
        """
        The number of pixels, not including depth
        """
        return self.height * self.width

_shape_queue = [] # type: List[WebGLArrayTextureShape]

def enqueue_default_texture_shape(*texture_shapes: WebGLArrayTextureShape):
    """
    Fixes the WebGLArrayTextureShape obtained the next time get_default_texture_shape is called to a specific value.
    """
    _shape_queue.extend(texture_shapes)

def get_default_texture_shape(size: int, dtype: np.dtype) -> WebGLArrayTextureShape:
    if len(_shape_queue) > 0:
        return _shape_queue.pop(0)
    if dtype == np.float32:
        if get_float_texture_bit() == 16:
            f = {
                    'internal_format': WebGL2RenderingContext.R16F,
                    'format': WebGL2RenderingContext.RED,
                    'type': WebGL2RenderingContext.HALF_FLOAT
                }
        else:
            f = {
                    'internal_format': WebGL2RenderingContext.R32F,
                    'format': WebGL2RenderingContext.RED,
                    'type': WebGL2RenderingContext.FLOAT
                }
    elif dtype == np.int32:
        f = {
                'internal_format': WebGL2RenderingContext.R32I,
                'format': WebGL2RenderingContext.RED_INTEGER,
                'type': WebGL2RenderingContext.INT
            }
    elif dtype == np.uint8:
        f = {
                'internal_format': WebGL2RenderingContext.R8UI,
                'format': WebGL2RenderingContext.RED_INTEGER,
                'type': WebGL2RenderingContext.UNSIGNED_BYTE
            }
    elif dtype == np.bool_:
        f = {
                'internal_format': WebGL2RenderingContext.R8UI,
                'format': WebGL2RenderingContext.RED_INTEGER,
                'type': WebGL2RenderingContext.UNSIGNED_BYTE
            }
    else:
        raise ValueError(f'WebGL: unsupported dtype {dtype}')
    
    # h, w
    mts = get_max_texture_size()
    dim = '2D'
    if size < mts:
        height = 1
        depth = 1
        width = max(size, 1) # avoid size 0
    else:
        width = mts
        height = int(math.ceil(size / mts))
        depth = 1
        if height > mts:
            dim = '2DArray'
            depth = int(math.ceil(height / mts))
            height = mts
            if depth > 16: # TODO
                raise ValueError('Array too large for WebGL texture')
    return WebGLArrayTextureShape(height=height, width=width, depth=depth, dim=dim, **f)
