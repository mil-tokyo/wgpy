from typing import List, Optional, Tuple, Union
import numpy as np
from wgpy_backends.webgpu.platform import get_platform
from wgpy_backends.webgpu.ndarray import ndarray
from wgpy_backends.webgpu.webgpu_buffer import WebGPUBuffer

added_kernels = set()
elementwise_kernels = {}

class MockUfunc:
    def __init__(self):
        pass

    def gt(self, lhs: ndarray, rhs: float) -> ndarray:
        # mock implementation for lhs > 0.0
        assert rhs == 0.0
        kernel_name = "gt"
        if kernel_name not in added_kernels:
            get_platform().addKernel(kernel_name, {
                'source': '''
@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read_write> array_c: array<i32>;

struct CMeta {
  K: i32,
}

@group(0) @binding(2)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(64,1,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var K: u32 = u32(cmeta.K);
  var x: u32 = global_id.x;
  for(var k: u32 = x; k < K; k = k + 4096u) {
    var v: f32 = array_a[k];
    var u: i32 = 0;
    if (v > 0.0) {
      u = 1;
    }
    array_c[k] = u;
  }
}
''',
                'bindingTypes': ['read-only-storage', 'storage', 'read-only-storage'],
            })
            added_kernels.add(kernel_name)
        
        meta = WebGPUBuffer(1, np.dtype(np.int32))
        meta.set_data(np.array([lhs.size], dtype=np.int32))
        array_c = ndarray(lhs.shape, np.bool_)

        get_platform().runKernel({
            'name': kernel_name,
            'tensors': [lhs.buffer.buffer_id, array_c.buffer.buffer_id, meta.buffer_id],
            'workGroups': {'x': 64, 'y': 1, 'z': 1},
        })
        return array_c

class WebGPUArrayFunc:
    _instance = None

    def __init__(self) -> None:
        self.ufunc = MockUfunc() # common_ufunc
        # self.reduction = common_reduction
        pass

    @staticmethod
    def instance() -> 'WebGPUArrayFunc':
        if WebGPUArrayFunc._instance is None:
            WebGPUArrayFunc._instance = WebGPUArrayFunc()
        return WebGPUArrayFunc._instance

    def matmul(self, lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> ndarray:
        # 2D only
        # TODO: irregular texture
        m, k = lhs.shape
        k2, n = rhs.shape
        assert k == k2
        for in_array in [lhs, rhs]:
            assert in_array.buffer.texture_shape.dim == '2D'
            assert in_array.buffer.texture_shape.internal_format in (WebGL2RenderingContext.R32F, WebGL2RenderingContext.R16F)
            assert in_array.buffer.texture_shape.format == WebGL2RenderingContext.RED
            assert in_array.buffer.texture_shape.type in (WebGL2RenderingContext.FLOAT, WebGL2RenderingContext.HALF_FLOAT)
        if out is not None:
            assert out.flags.c_contiguous_full
            assert out.buffer.texture_shape.dim == '2D'
            assert out.buffer.texture_shape.internal_format in (WebGL2RenderingContext.R32F, WebGL2RenderingContext.R16F)
            assert out.buffer.texture_shape.format == WebGL2RenderingContext.RED
            assert out.buffer.texture_shape.type in (WebGL2RenderingContext.FLOAT, WebGL2RenderingContext.HALF_FLOAT)
            for in_array in [lhs, rhs]:
                assert out.buffer.buffer_id != in_array.buffer.buffer_id
            assert out.shape == (m, n)
        kernel_name = f'matmul_{m}_{n}_{k}'
        if kernel_name not in added_kernels:
            get_platform().addKernel(kernel_name, {'source': f"""#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
precision highp isampler2D;
precision highp isampler2DArray;
precision highp usampler2D;
precision highp usampler2DArray;

#define M {m}
#define N {n}
#define K {k}

uniform int _ka_tex_output_texture_w;
uniform int LHS_STRIDE_0;
uniform int LHS_STRIDE_1;
uniform int RHS_STRIDE_0;
uniform int RHS_STRIDE_1;
uniform sampler2D tex_lhs;
uniform sampler2D tex_rhs;

out float fragColor;

float get_tex_lhs(int d0, int d1) {{
int flat_index = d0 * LHS_STRIDE_0 + d1 * LHS_STRIDE_1;
int texture_w = textureSize(tex_lhs, 0).x;
int y = flat_index / texture_w;
int x = flat_index - y * texture_w;
return texelFetch(tex_lhs, ivec2(x, y), 0).r;
}}

float get_tex_rhs(int d0, int d1) {{
int flat_index = d0 * RHS_STRIDE_0 + d1 * RHS_STRIDE_1;
int texture_w = textureSize(tex_rhs, 0).x;
int y = flat_index / texture_w;
int x = flat_index - y * texture_w;
return texelFetch(tex_rhs, ivec2(x, y), 0).r;
}}


void main() {{
    int flat_idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
    int oi = flat_idx / N;
    int oj = flat_idx - oi * N;
    if (oi >= M) {{ return; }}
    float s = 0.0;
    for (int k = 0; k < K; k++) {{
        s += get_tex_lhs(oi, k) * get_tex_rhs(k, oj);
    }}
    fragColor = s;
}}
    """})
            added_kernels.add(kernel_name)
        if out is None:
            out = ndarray((m, n), lhs.dtype)
        get_platform().runKernel({'name': kernel_name, 'inputs': [{'name': 'tex_lhs', 'id': lhs.buffer.buffer_id}, {
                     'name': 'tex_rhs', 'id': rhs.buffer.buffer_id}], 'output': out.buffer.buffer_id, 'uniforms': [
                        {'name': '_ka_tex_output_texture_w', 'value': out.buffer.texture_shape.width, 'type': 'int'},
                        {'name': 'LHS_STRIDE_0', 'value': lhs.strides[0] // lhs.itemsize, 'type': 'int'},
                        {'name': 'LHS_STRIDE_1', 'value': lhs.strides[1] // lhs.itemsize, 'type': 'int'},
                        {'name': 'RHS_STRIDE_0', 'value': rhs.strides[0] // rhs.itemsize, 'type': 'int'},
                        {'name': 'RHS_STRIDE_1', 'value': rhs.strides[1] // rhs.itemsize, 'type': 'int'}
                        ]})
        return out
