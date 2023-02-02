import numpy as np
from wgpy_backends.webgl.texture import WebGL2RenderingContext

header = """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp isampler2D;
precision highp isampler2DArray;
precision highp usampler2D;
precision highp usampler2DArray;
"""

native_scalar_type_for_type = {
    WebGL2RenderingContext.FLOAT: 'float',
    WebGL2RenderingContext.HALF_FLOAT: 'float',
    WebGL2RenderingContext.INT: 'int',
    WebGL2RenderingContext.UNSIGNED_BYTE: 'uint',
}

native_scalar_type_for_dtype = {
    np.dtype(np.float32): 'float',
    np.dtype(np.int32): 'int',
    np.dtype(np.uint8): 'uint',
    np.dtype(np.bool_): 'bool',
}

native_scalar_type_to_default_dtype = {
    'float': np.dtype(np.float32),
    'int': np.dtype(np.int32),
    'uint': np.dtype(np.uint8),
    'bool': np.dtype(np.bool_),
}

native_pixel_type_for_internal_format = {
    WebGL2RenderingContext.R32F: 'float',
    WebGL2RenderingContext.R16F: 'float',
    WebGL2RenderingContext.R32I: 'int',
    WebGL2RenderingContext.R8UI: 'uint',
    WebGL2RenderingContext.RGBA32F: 'vec4',
    WebGL2RenderingContext.RGBA16F: 'vec4',
    WebGL2RenderingContext.RGBA32I: 'ivec4',
    WebGL2RenderingContext.RGBA8UI: 'uvec4',
}

