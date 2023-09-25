import math
from typing import List, Optional, Tuple, Union
from wgpy_backends.webgpu import common_reduction
from wgpy_backends.webgpu import common_ufunc
from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
from wgpy_backends.webgpu.platform import get_platform
from wgpy_backends.webgpu.ndarray import ndarray

added_kernels = set()

def _matmul_generic(lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> ndarray:
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
LHS_OFFSET: u32,
LHS_STRIDE_0: u32,
LHS_STRIDE_1: u32,
RHS_OFFSET: u32,
RHS_STRIDE_0: u32,
RHS_STRIDE_1: u32,
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
var LHS_OFFSET: u32 = cmeta.LHS_OFFSET;
var LHS_STRIDE_0: u32 = cmeta.LHS_STRIDE_0;
var LHS_STRIDE_1: u32 = cmeta.LHS_STRIDE_1;
var RHS_OFFSET: u32 = cmeta.RHS_OFFSET;
var RHS_STRIDE_0: u32 = cmeta.RHS_STRIDE_0;
var RHS_STRIDE_1: u32 = cmeta.RHS_STRIDE_1;
var x: u32 = global_id.x;
var y: u32 = global_id.y;
if (x >= N || y >= M) {
return;
}
var sum: f32 = 0.0;
for(var k: u32 = 0u; k < K; k = k + 1u) {
sum = array_a[LHS_OFFSET + y * LHS_STRIDE_0 + k * LHS_STRIDE_1] * array_b[RHS_OFFSET + k * RHS_STRIDE_0 + x * RHS_STRIDE_1] + sum;
}
array_c[x + y * N] = sum * cmeta.alpha;
}
""",
            'bindingTypes': ['read-only-storage', 'read-only-storage', 'storage', 'read-only-storage'],})
        added_kernels.add(kernel_name)
    meta = create_meta_buffer_from_structure((m, n, k, lhs.offset // lhs.itemsize, lhs.strides[0] // lhs.itemsize, lhs.strides[1] // lhs.itemsize, rhs.offset // rhs.itemsize, rhs.strides[0] // rhs.itemsize, rhs.strides[1] // rhs.itemsize, 1.0), 'u4,u4,u4,u4,u4,u4,u4,u4,u4,f4')
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

def _matmul_m32n64k4_check(lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> bool:
    m, k = lhs.shape
    _, n = rhs.shape
    return m % 32 == 0 and n % 64 == 0 and k % 4 == 0 \
        and lhs.flags.c_contiguous_full and rhs.flags.c_contiguous_full \
        and (out is None or out.flags.c_contiguous_full)

def _matmul_m32n64k4(lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> ndarray:
    m, k = lhs.shape
    _, n = rhs.shape
    kernel_name = f'matmul_m32n64k4'
    if kernel_name not in added_kernels:
        get_platform().addKernel(kernel_name, {'source': """@group(0) @binding(0)
var<storage,read> array_a: array<vec4<f32>>;

@group(0) @binding(1)
var<storage,read> array_b: array<vec4<f32>>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<vec4<f32>>;

struct CMeta {
M: u32,
N: u32,
K: u32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(8,8,1)
fn main(
@builtin(global_invocation_id) global_id: vec3<u32>
) {
let M: u32 = cmeta.M;
let N: u32 = cmeta.N;
let K: u32 = cmeta.K;
let MD4: u32 = M >> 2;
let ND4: u32 = N >> 2;
let KD4: u32 = K >> 2;
var x: u32 = global_id.x;
var y: u32 = global_id.y;
if (x >= N || y >= M) {
return;
}
var sum00: vec4<f32> = vec4<f32>();
var sum01: vec4<f32> = vec4<f32>();
var sum02: vec4<f32> = vec4<f32>();
var sum03: vec4<f32> = vec4<f32>();
var sum10: vec4<f32> = vec4<f32>();
var sum11: vec4<f32> = vec4<f32>();
var sum12: vec4<f32> = vec4<f32>();
var sum13: vec4<f32> = vec4<f32>();
for(var k: u32 = 0u; k < KD4; k = k + 1u) {
var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
var brow: vec4<f32>;
brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
sum00 = vec4<f32>(arow0.x) * brow + sum00;
sum01 = vec4<f32>(arow1.x) * brow + sum01;
sum02 = vec4<f32>(arow2.x) * brow + sum02;
sum03 = vec4<f32>(arow3.x) * brow + sum03;
brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
sum10 = vec4<f32>(arow0.x) * brow + sum10;
sum11 = vec4<f32>(arow1.x) * brow + sum11;
sum12 = vec4<f32>(arow2.x) * brow + sum12;
sum13 = vec4<f32>(arow3.x) * brow + sum13;

brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
sum00 = vec4<f32>(arow0.y) * brow + sum00;
sum01 = vec4<f32>(arow1.y) * brow + sum01;
sum02 = vec4<f32>(arow2.y) * brow + sum02;
sum03 = vec4<f32>(arow3.y) * brow + sum03;
brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
sum10 = vec4<f32>(arow0.y) * brow + sum10;
sum11 = vec4<f32>(arow1.y) * brow + sum11;
sum12 = vec4<f32>(arow2.y) * brow + sum12;
sum13 = vec4<f32>(arow3.y) * brow + sum13;

brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
sum00 = vec4<f32>(arow0.z) * brow + sum00;
sum01 = vec4<f32>(arow1.z) * brow + sum01;
sum02 = vec4<f32>(arow2.z) * brow + sum02;
sum03 = vec4<f32>(arow3.z) * brow + sum03;
brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
sum10 = vec4<f32>(arow0.z) * brow + sum10;
sum11 = vec4<f32>(arow1.z) * brow + sum11;
sum12 = vec4<f32>(arow2.z) * brow + sum12;
sum13 = vec4<f32>(arow3.z) * brow + sum13;

brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
sum00 = vec4<f32>(arow0.w) * brow + sum00;
sum01 = vec4<f32>(arow1.w) * brow + sum01;
sum02 = vec4<f32>(arow2.w) * brow + sum02;
sum03 = vec4<f32>(arow3.w) * brow + sum03;
brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
sum10 = vec4<f32>(arow0.w) * brow + sum10;
sum11 = vec4<f32>(arow1.w) * brow + sum11;
sum12 = vec4<f32>(arow2.w) * brow + sum12;
sum13 = vec4<f32>(arow3.w) * brow + sum13;
}
array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00;
array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01;
array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02;
array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03;
array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10;
array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11;
array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12;
array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13;
}
""",
            'bindingTypes': ['read-only-storage', 'read-only-storage', 'storage', 'read-only-storage'],})
        added_kernels.add(kernel_name)
    meta = create_meta_buffer_from_structure((m, n, k), 'u4,u4,u4')
    if out is None:
        out = ndarray((m, n), lhs.dtype)
    else:
        assert out.flags.c_contiguous_full

    get_platform().runKernel({
        'name': kernel_name,
        'tensors': [lhs.buffer.buffer_id, rhs.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
        'workGroups': {'x': int(n//64), 'y': int(m//32), 'z': 1},
    })

    return out

def matmul_impl(lhs: ndarray, rhs: ndarray, out: Optional[ndarray]=None) -> ndarray:
    # 2D only
    m, k = lhs.shape
    k2, n = rhs.shape
    assert k == k2
    if _matmul_m32n64k4_check(lhs, rhs, out):
        return _matmul_m32n64k4(lhs, rhs, out)
    return _matmul_generic(lhs, rhs, out)
