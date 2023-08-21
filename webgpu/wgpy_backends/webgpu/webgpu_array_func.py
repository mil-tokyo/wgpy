import math
from typing import Optional
from wgpy_backends.webgpu import common_ufunc
from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
from wgpy_backends.webgpu.platform import get_platform
from wgpy_backends.webgpu.ndarray import ndarray

added_kernels = set()
elementwise_kernels = {}


class WebGPUArrayFunc:
    _instance = None

    def __init__(self) -> None:
        self.ufunc = common_ufunc
        # self.reduction = common_reduction
        pass

    @staticmethod
    def instance() -> 'WebGPUArrayFunc':
        if WebGPUArrayFunc._instance is None:
            WebGPUArrayFunc._instance = WebGPUArrayFunc()
        return WebGPUArrayFunc._instance

    def matmul(self, lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> ndarray:
        # 2D only
        # TODO: accelerate in special case (such as k is multiple of 4)
        m, k = lhs.shape
        k2, n = rhs.shape
        assert k == k2
        kernel_name = f'matmul'
        if kernel_name not in added_kernels:
            get_platform().addKernel(kernel_name, {'source': """@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
  M: u32,
  N: u32,
  K: u32,
  alpha: f32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var M: u32 = cmeta.M;
  var N: u32 = cmeta.N;
  var K: u32 = cmeta.K;
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }
  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k = k + 1u) {
    sum = array_a[y * K + k] * array_b[k * N + x] + sum;
  }
  array_c[x + y * N] = sum * cmeta.alpha;
}
    """,
                'bindingTypes': ['read-only-storage', 'read-only-storage', 'storage', 'read-only-storage'],})
            added_kernels.add(kernel_name)
        # TODO: consider strides
        assert lhs.flags.c_contiguous_full
        assert rhs.flags.c_contiguous_full
        meta = create_meta_buffer_from_structure((m, n, k, 1.0), 'u4,u4,u4,f4')
        if out is None:
            out = ndarray((m, n), lhs.dtype)
        else:
            assert out.flags.c_contiguous_full

        get_platform().runKernel({
            'name': kernel_name,
            'tensors': [lhs.buffer.buffer_id, rhs.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
            'workGroups': {'x': int(math.ceil(n/8)), 'y': int(math.ceil(m/8)), 'z': 1},
        })

        return out
