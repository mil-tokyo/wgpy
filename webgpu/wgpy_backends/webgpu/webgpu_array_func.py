import math
from typing import List, Optional, Tuple, Union
import numpy as np
from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
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
  K: u32,
}

@group(0) @binding(2)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(64,1,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var K: u32 = cmeta.K;
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
        
        meta = create_meta_buffer_from_structure((lhs.size,), 'u4')
        array_c = ndarray(lhs.shape, np.bool_)

        get_platform().runKernel({
            'name': kernel_name,
            'tensors': [lhs.buffer.buffer_id, array_c.buffer.buffer_id, meta.buffer_id],
            'workGroups': {'x': 64, 'y': 1, 'z': 1},
        })
        return array_c
    
    def _add_1d_f32(self, lhs: ndarray, rhs: ndarray) -> ndarray:
        kernel_name = "add_1d_f32"
        if kernel_name not in added_kernels:
            get_platform().addKernel(kernel_name, {
                'source': '''
struct CMeta {
  out_shape: array<i32, 1>,
  in0_offset: i32,
  in0_strides: array<i32, 1>,
  in1_offset: i32,
  in1_strides: array<i32, 1>,
}

@group(0) @binding(0)
var<storage,read> cmeta: CMeta;

@group(0) @binding(1)
var<storage,read> array_in0: array<f32>;

@group(0) @binding(2)
var<storage,read> array_in1: array<f32>;

@group(0) @binding(3)
var<storage,read_write> array_out: array<f32>;

fn _in0(idx0: i32) -> f32 {
  return array_in0[cmeta.in0_offset + idx0 * cmeta.in0_strides[0]];
}


fn _in1(idx0: i32) -> f32 {
  return array_in1[cmeta.in1_offset + idx0 * cmeta.in1_strides[0]];
}

@compute @workgroup_size(64,1,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  for(var k: i32 = i32(global_id.x);; k = k + 4096i) {
    var _out0_0 = k;
    if (_out0_0 > cmeta.out_shape[0]) {
      break;
    }
    var v_in0 = _in0(_out0_0);
    var v_in1 = _in1(_out0_0);
    var v_out = v_in0 + v_in1;
    array_out[k] = v_out;
  }
}
''',
                'bindingTypes': ['read-only-storage', 'read-only-storage', 'read-only-storage', 'storage'],
            })
            added_kernels.add(kernel_name)
        
        assert lhs.shape == rhs.shape
        array_c = ndarray(lhs.shape, lhs.dtype)
        meta = create_meta_buffer_from_structure((
            array_c.shape[0],
            lhs.offset // lhs.itemsize,
            lhs.strides[0] // lhs.itemsize,
            rhs.offset // rhs.itemsize,
            rhs.strides[0] // rhs.itemsize,
            ), 'i4,i4,i4,i4,i4')

        get_platform().runKernel({
            'name': kernel_name,
            'tensors': [meta.buffer_id, lhs.buffer.buffer_id, rhs.buffer.buffer_id, array_c.buffer.buffer_id],
            'workGroups': {'x': 64, 'y': 1, 'z': 1},
        })
        return array_c

    def add(self, lhs: ndarray, rhs: ndarray) -> ndarray:
        if isinstance(lhs, ndarray) and isinstance(rhs, ndarray):
            if lhs.dtype == np.float32 and rhs.dtype == np.float32:
                if lhs.ndim == 1 and rhs.ndim == 1:
                  return self._add_1d_f32(lhs, rhs)
        raise NotImplementedError(f'add is not implemented for lhs={lhs} and rhs={rhs}')

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
