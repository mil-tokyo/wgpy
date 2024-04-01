import math
from typing import List, Optional, Tuple, Union
from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
from wgpy_backends.webgpu.platform import get_platform
from wgpy_backends.webgpu.ndarray import ndarray

added_kernels = set()
elementwise_kernels = {}


def _matmul_generic(
    lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None
) -> ndarray:
    m, k = lhs.shape
    k2, n = rhs.shape
    assert k == k2
    kernel_name = f"matmul"
    if kernel_name not in added_kernels:
        get_platform().addKernel(
            kernel_name,
            {
                "source": """@group(0) @binding(0)
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
                "bindingTypes": [
                    "read-only-storage",
                    "read-only-storage",
                    "storage",
                    "read-only-storage",
                ],
            },
        )
        added_kernels.add(kernel_name)
    meta = create_meta_buffer_from_structure(
        (
            m,
            n,
            k,
            lhs.offset // lhs.itemsize,
            lhs.strides[0] // lhs.itemsize,
            lhs.strides[1] // lhs.itemsize,
            rhs.offset // rhs.itemsize,
            rhs.strides[0] // rhs.itemsize,
            rhs.strides[1] // rhs.itemsize,
            1.0,
        ),
        "u4,u4,u4,u4,u4,u4,u4,u4,u4,f4",
    )
    if out is None:
        out = ndarray((m, n), lhs.dtype)
    else:
        assert out.flags.c_contiguous_full

    get_platform().runKernel(
        {
            "name": kernel_name,
            "tensors": [
                lhs.buffer.buffer_id,
                rhs.buffer.buffer_id,
                out.buffer.buffer_id,
                meta.buffer_id,
            ],
            "workGroups": {
                "x": int(math.ceil(n / 8)),
                "y": int(math.ceil(m / 8)),
                "z": 1,
            },
        }
    )

    return out


def _matmul_m32n64k4_check(
    lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None
) -> bool:
    m, k = lhs.shape
    _, n = rhs.shape
    return (
        m % 32 == 0
        and n % 64 == 0
        and k % 4 == 0
        and lhs.flags.c_contiguous_full
        and rhs.flags.c_contiguous_full
        and (out is None or out.flags.c_contiguous_full)
    )


def _matmul_m32n64k4(
    lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None
) -> ndarray:
    m, k = lhs.shape
    _, n = rhs.shape
    kernel_name = f"matmul_m32n64k4"
    if kernel_name not in added_kernels:
        get_platform().addKernel(
            kernel_name,
            {
                "source": """@group(0) @binding(0)
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
                "bindingTypes": [
                    "read-only-storage",
                    "read-only-storage",
                    "storage",
                    "read-only-storage",
                ],
            },
        )
        added_kernels.add(kernel_name)
    meta = create_meta_buffer_from_structure((m, n, k), "u4,u4,u4")
    if out is None:
        out = ndarray((m, n), lhs.dtype)
    else:
        assert out.flags.c_contiguous_full

    get_platform().runKernel(
        {
            "name": kernel_name,
            "tensors": [
                lhs.buffer.buffer_id,
                rhs.buffer.buffer_id,
                out.buffer.buffer_id,
                meta.buffer_id,
            ],
            "workGroups": {"x": int(n // 64), "y": int(m // 32), "z": 1},
        }
    )

    return out


def matmul_impl(lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None) -> ndarray:
    # 2D only
    m, k = lhs.shape
    k2, n = rhs.shape
    assert k == k2
    if _matmul_m32n64k4_check(lhs, rhs, out):
        return _matmul_m32n64k4(lhs, rhs, out)
    return _matmul_generic(lhs, rhs, out)


def _tensordot_convforward_check(
    a: ndarray, b: ndarray, axes: Tuple[List[int], List[int]]
) -> bool:
    if a.ndim != 6 or b.ndim != 4:
        return False
    if len(axes) != 2:
        return False
    if list(axes[0]) != [1, 2, 3] or list(axes[1]) != [1, 2, 3]:
        return False
    if not a.flags.c_contiguous_full or not b.flags.c_contiguous_full:
        return False
    n, c, kh, kw, h, w = a.shape
    if h * w % 4 != 0:
        return False
    oc, _, _, _ = b.shape
    if oc % 4 != 0:
        return False
    return True


def _tensordot_convforward(a: ndarray, b: ndarray) -> ndarray:
    # convolution forward: tensordot(col(ndim=6), W(ndim=4), axes=((1,2,3),(1,2,3)))
    # a: (N, C, KH, KW, H, W) treat as (N, C*KH*KW, H*W)
    # b: (OC, C, KH, KW) treat as (OC, C*KH*KW)
    # c: (N, H, W, OC) treat as (N, H*W, OC)
    n, c, kh, kw, h, w = a.shape
    oc, _, _, _ = b.shape
    ckhkw = c * kh * kw
    hw = h * w
    hw_is_mul_32 = hw % 32 == 0
    oc_is_mul_32 = oc % 32 == 0
    kernel_key = ("tensordot_cf", hw_is_mul_32, oc_is_mul_32)
    kernel = elementwise_kernels.get(kernel_key)
    kernel_name = repr(kernel_key)
    if kernel is None:
        get_platform().addKernel(
            kernel_name,
            {
                "source": """@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
k: u32,
ast0: u32,
ast1: u32,
ast2: u32,
bst0: u32,
bst1: u32,
cst0: u32,
cst1: u32,
cst2: u32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

"""
                f"@compute @workgroup_size(1,{8 if hw_is_mul_32 else 1},{8 if oc_is_mul_32 else 1})"
                """
fn main(
@builtin(global_invocation_id) global_id: vec3<u32>
) {
// a: (N, C*KH*KW, H*W)
// b: (OC, C*KH*KW)
// c: (N, H*W, OC)
// output index global_id.x: a axis=0, y: a axis=2 / 4, z: b axis=0 / 4
var s: mat4x4<f32> = mat4x4<f32>();
for (var k: u32 = 0; k < cmeta.k; k = k + 1u) {
    let aofs = cmeta.ast0 * global_id.x + cmeta.ast1 * k + cmeta.ast2 * (global_id.y * 4);
    let avals = vec4<f32>(array_a[aofs], array_a[aofs + cmeta.ast2 * 1], array_a[aofs + cmeta.ast2 * 2], array_a[aofs + cmeta.ast2 * 3]);
    let bofs = cmeta.bst0 * (global_id.z * 4) + cmeta.bst1 * k;
    let bvals = vec4<f32>(array_b[bofs], array_b[bofs + cmeta.bst0 * 1], array_b[bofs + cmeta.bst0 * 2], array_b[bofs + cmeta.bst0 * 3]);
    s[0] = s[0] + avals * bvals[0];
    s[1] = s[1] + avals * bvals[1];
    s[2] = s[2] + avals * bvals[2];
    s[3] = s[3] + avals * bvals[3];
}
let cofs = cmeta.cst0 * global_id.x + cmeta.cst1 * (global_id.y * 4) + cmeta.cst2 * (global_id.z * 4);
for (var r: u32 = 0; r < 4; r = r + 1u) {
    for (var c: u32 = 0; c < 4; c = c + 1u) {
        array_c[cofs + cmeta.cst1 * r + cmeta.cst2 * c] = s[c][r];
    }
}
}
""",
                "bindingTypes": [
                    "read-only-storage",
                    "read-only-storage",
                    "storage",
                    "read-only-storage",
                ],
            },
        )
        added_kernels.add(kernel_name)

    out = ndarray((n, h, w, oc), a.dtype)
    meta = create_meta_buffer_from_structure(
        (
            ckhkw,
            ckhkw * hw,
            hw,
            1,
            ckhkw,
            1,
            hw * oc,
            oc,
            1,
        ),
        "u4,u4,u4,u4,u4,u4,u4,u4,u4",
    )

    get_platform().runKernel(
        {
            "name": kernel_name,
            "tensors": [
                a.buffer.buffer_id,
                b.buffer.buffer_id,
                out.buffer.buffer_id,
                meta.buffer_id,
            ],
            "workGroups": {
                "x": int(n),
                "y": int(hw // 32 if hw_is_mul_32 else hw // 4),
                "z": int(oc // 32 if oc_is_mul_32 else oc // 4),
            },
        }
    )

    return out


def _tensordot_convbackwardinput_check(
    a: ndarray, b: ndarray, axes: Tuple[List[int], List[int]]
) -> bool:
    if a.ndim != 4 or b.ndim != 4:
        return False
    if len(axes) != 2:
        return False
    if list(axes[0]) != [0] or list(axes[1]) != [1]:
        return False
    if not a.flags.c_contiguous_full or not b.flags.c_contiguous_full:
        return False
    oc, c, kh, kw = a.shape
    n, _, h, w = b.shape
    if c * kh * kw % 4 != 0:
        return False
    if h * w % 4 != 0:
        return False
    return True


def _tensordot_convbackwardinput(a: ndarray, b: ndarray) -> ndarray:
    # convolution backward col: tensordot(W(ndim=4), x(ndim=4), axes=(0, 1))
    # a: (OC, C, KH, KW) treat as (OC, C*KH*KW)
    # b: (N, OC, H, W) treat as (N, OC, H*W)
    # c: (C, KH, KW, N, H, W) treat as (C*KH*KW, N, H*W)
    oc, c, kh, kw = a.shape
    n, _, h, w = b.shape
    ckhkw = c * kh * kw
    hw = h * w
    ckhkw_is_mul_32 = ckhkw % 32 == 0
    hw_is_mul_32 = hw % 32 == 0
    kernel_key = ("tensordot_cbi", ckhkw_is_mul_32, hw_is_mul_32)
    kernel = elementwise_kernels.get(kernel_key)
    kernel_name = repr(kernel_key)
    if kernel is None:
        get_platform().addKernel(
            kernel_name,
            {
                "source": """@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
k: u32,
ast0: u32,
ast1: u32,
bst0: u32,
bst1: u32,
bst2: u32,
cst0: u32,
cst1: u32,
cst2: u32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

"""
                f"@compute @workgroup_size({8 if ckhkw_is_mul_32 else 1},1,{8 if hw_is_mul_32 else 1})"
                """
fn main(
@builtin(global_invocation_id) global_id: vec3<u32>
) {
// a: (OC, C*KH*KW)
// b: (N, OC, H*W)
// c: (C*KH*KW, N, H*W)
// output index global_id.x: a axis=1 / 4, y: b axis=0, z: b axis=2 / 4
var s: mat4x4<f32> = mat4x4<f32>();
for (var k: u32 = 0; k < cmeta.k; k = k + 1u) {
    let aofs = cmeta.ast0 * k + cmeta.ast1 * (global_id.x * 4);
    let avals = vec4<f32>(array_a[aofs], array_a[aofs + cmeta.ast1 * 1], array_a[aofs + cmeta.ast1 * 2], array_a[aofs + cmeta.ast1 * 3]);
    let bofs = cmeta.bst0 * global_id.y + cmeta.bst1 * k + cmeta.bst2 * (global_id.z * 4);
    let bvals = vec4<f32>(array_b[bofs], array_b[bofs + cmeta.bst2 * 1], array_b[bofs + cmeta.bst2 * 2], array_b[bofs + cmeta.bst2 * 3]);
    s[0] = s[0] + avals * bvals[0];
    s[1] = s[1] + avals * bvals[1];
    s[2] = s[2] + avals * bvals[2];
    s[3] = s[3] + avals * bvals[3];
}
let cofs = cmeta.cst0 * (global_id.x * 4) + cmeta.cst1 * global_id.y + cmeta.cst2 * (global_id.z * 4);
for (var r: u32 = 0; r < 4; r = r + 1u) {
    for (var c: u32 = 0; c < 4; c = c + 1u) {
        array_c[cofs + cmeta.cst0 * r + cmeta.cst2 * c] = s[c][r];
    }
}
}
""",
                "bindingTypes": [
                    "read-only-storage",
                    "read-only-storage",
                    "storage",
                    "read-only-storage",
                ],
            },
        )
        added_kernels.add(kernel_name)

    out = ndarray((c, kh, kw, n, h, w), a.dtype)
    meta = create_meta_buffer_from_structure(
        (
            oc,
            ckhkw,
            1,
            oc * hw,
            hw,
            1,
            n * hw,
            hw,
            1,
        ),
        "u4,u4,u4,u4,u4,u4,u4,u4,u4",
    )

    get_platform().runKernel(
        {
            "name": kernel_name,
            "tensors": [
                a.buffer.buffer_id,
                b.buffer.buffer_id,
                out.buffer.buffer_id,
                meta.buffer_id,
            ],
            "workGroups": {
                "x": int(ckhkw // 32 if ckhkw_is_mul_32 else ckhkw // 4),
                "y": int(n),
                "z": int(hw // 32 if hw_is_mul_32 else hw // 4),
            },
        }
    )

    return out


def _tensordot_convbackwardweight_check(
    a: ndarray, b: ndarray, axes: Tuple[List[int], List[int]]
) -> bool:
    if a.ndim != 4 or b.ndim != 6:
        return False
    if len(axes) != 2:
        return False
    if list(axes[0]) != [0, 2, 3] or list(axes[1]) != [0, 4, 5]:
        return False
    if not a.flags.c_contiguous_full or not b.flags.c_contiguous_full:
        return False
    n, oc, h, w = a.shape
    _, c, kh, kw, _, _ = b.shape
    if c * kh * kw % 4 != 0:
        return False
    if oc % 4 != 0:
        return False
    return True


def _tensordot_convbackwardweight(a: ndarray, b: ndarray) -> ndarray:
    # convolution backward weight: tensordot(gy(ndim=4), col(ndim=6), axes=((0,2,3),(0,4,5)))
    # a: (N, OC, H, W) treat as (N, OC, H*W)
    # b: (N, C, KH, KW, H, W) treat as (N, C*KH*KW, H*W)
    # c: (OC, C, KH, KW) treat as (OC, C*KH*KW)
    n, oc, h, w = a.shape
    _, c, kh, kw, _, _ = b.shape
    ckhkw = c * kh * kw
    hw = h * w
    oc_is_mul_32 = oc % 32 == 0
    ckhkw_is_mul_32 = ckhkw % 32 == 0
    kernel_key = ("tensordot_cbw", oc_is_mul_32, ckhkw_is_mul_32)
    kernel = elementwise_kernels.get(kernel_key)
    kernel_name = repr(kernel_key)
    if kernel is None:
        get_platform().addKernel(
            kernel_name,
            {
                "source": """@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
k0: u32, // for N
k1: u32, // for H*W
ast0: u32,
ast1: u32,
ast2: u32,
bst0: u32,
bst1: u32,
bst2: u32,
cst0: u32,
cst1: u32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

"""
                f"@compute @workgroup_size({8 if oc_is_mul_32 else 1},{8 if ckhkw_is_mul_32 else 1},1)"
                """
fn main(
@builtin(global_invocation_id) global_id: vec3<u32>
) {
// a: (N, OC, H*W)
// b: (N, C*KH*KW, H*W)
// c: (OC, C*KH*KW)
// output index global_id.x: a axis=1 / 4, y: b axis=1 / 4, z = 0
var s: mat4x4<f32> = mat4x4<f32>();
for (var k0: u32 = 0; k0 < cmeta.k0; k0 = k0 + 1u) {
    for (var k1: u32 = 0; k1 < cmeta.k1; k1 = k1 + 1u) {
        let aofs = cmeta.ast0 * k0 + cmeta.ast1 * (global_id.x * 4) + cmeta.ast2 * k1;
        let avals = vec4<f32>(array_a[aofs], array_a[aofs + cmeta.ast1 * 1], array_a[aofs + cmeta.ast1 * 2], array_a[aofs + cmeta.ast1 * 3]);
        let bofs = cmeta.bst0 * k0 + cmeta.bst1 * (global_id.y * 4) + cmeta.bst2 * k1;
        let bvals = vec4<f32>(array_b[bofs], array_b[bofs + cmeta.bst1 * 1], array_b[bofs + cmeta.bst1 * 2], array_b[bofs + cmeta.bst1 * 3]);
        s[0] = s[0] + avals * bvals[0];
        s[1] = s[1] + avals * bvals[1];
        s[2] = s[2] + avals * bvals[2];
        s[3] = s[3] + avals * bvals[3];
    }
}
let cofs = cmeta.cst0 * (global_id.x * 4) + cmeta.cst1 * (global_id.y * 4);
for (var r: u32 = 0; r < 4; r = r + 1u) {
    for (var c: u32 = 0; c < 4; c = c + 1u) {
        array_c[cofs + cmeta.cst0 * r + cmeta.cst1 * c] = s[c][r];
    }
}
}
""",
                "bindingTypes": [
                    "read-only-storage",
                    "read-only-storage",
                    "storage",
                    "read-only-storage",
                ],
            },
        )
        added_kernels.add(kernel_name)

    out = ndarray((oc, c, kh, kw), a.dtype)
    meta = create_meta_buffer_from_structure(
        (
            n,
            hw,
            oc * hw,
            hw,
            1,
            ckhkw * hw,
            hw,
            1,
            ckhkw,
            1,
        ),
        "u4,u4,u4,u4,u4,u4,u4,u4,u4,u4",
    )

    get_platform().runKernel(
        {
            "name": kernel_name,
            "tensors": [
                a.buffer.buffer_id,
                b.buffer.buffer_id,
                out.buffer.buffer_id,
                meta.buffer_id,
            ],
            "workGroups": {
                "x": int(oc // 32 if oc_is_mul_32 else oc // 4),
                "y": int(ckhkw // 32 if ckhkw_is_mul_32 else ckhkw // 4),
                "z": 1,
            },
        }
    )

    return out


def _unify_tensordot_axis(
    ndims: Tuple[int, int],
    axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]],
) -> Tuple[List[int], List[int]]:
    if isinstance(axes, int):
        # axes=2: ([ndims[0]-1,ndims[0]-2], [0, 1])
        return [[ndims[0] - i - 1 for i in range(axes)], [i for i in range(axes)]]
    if isinstance(axes[0], int):
        assert isinstance(axes[1], int)
        return ([axes[0]], [axes[1]])
    assert len(axes[0]) == len(axes[1])
    return axes


def _tensordot_generic(
    a: ndarray, b: ndarray, u_axes: Tuple[List[int], List[int]]
) -> ndarray:
    ip_shape = []
    for a_axis, b_axis in zip(u_axes[0], u_axes[1]):
        s = a.shape[a_axis]
        assert s == b.shape[b_axis]
        ip_shape.append(s)
    result_shape = []
    for dim in range(a.ndim):
        if dim not in u_axes[0]:
            result_shape.append(a.shape[dim])
    for dim in range(b.ndim):
        if dim not in u_axes[1]:
            result_shape.append(b.shape[dim])

    kernel_key = (
        "tensordot",
        a.ndim,
        b.ndim,
        tuple(u_axes[0]),
        tuple(u_axes[1]),
        tuple(ip_shape),
    )
    kernel = elementwise_kernels.get(kernel_key)
    if kernel is None:
        # when axes=([1,2],[1,0])
        # source = f'''
        # out0 = T(0);
        # for (int ip0 = 0; ip0 < IP0; ip0++) {{
        #     for (int ip1 = 0; ip1 < IP1; ip1++) {{
        #         out0 += a(_out0_0, ip0, ip1) * b(ip1, ip0, _out0_1);
        #     }}
        # }}
        # '''
        a_keys = []
        b_keys = []
        out_count = 0
        for dim in range(a.ndim):
            try:
                ip_idx = u_axes[0].index(dim)
                a_keys.append(f"ip{ip_idx}")
            except ValueError:
                a_keys.append(f"_out0_{out_count}")
                out_count += 1
        for dim in range(b.ndim):
            try:
                ip_idx = u_axes[1].index(dim)
                b_keys.append(f"ip{ip_idx}")
            except ValueError:
                b_keys.append(f"_out0_{out_count}")
                out_count += 1
        lf = "\n"
        source = f"""
{lf.join(f"const IP{dim}: i32 = {ip_shape[dim]};" for dim in range(len(ip_shape)))}

out0 = T(0);

{lf.join(f"for (var ip{dim}: i32 = 0; ip{dim} < IP{dim}; ip{dim}++){{" for dim in range(len(ip_shape)))}
out0 += a({','.join(a_keys)}) * b({','.join(b_keys)});
{lf.join(f"}}" for _ in range(len(ip_shape)))}

"""
        from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel

        kernel = ElementwiseKernel(
            in_params="rawnd T a, rawnd T b",
            out_params="T out0",
            operation=source,
            name="tensordot",
        )
        elementwise_kernels[kernel_key] = kernel
    from wgpy.construct import empty

    out = empty(tuple(result_shape), dtype=a.dtype)
    kernel(a, b, out)
    return out


def tensordot_impl(
    a: ndarray,
    b: ndarray,
    axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]],
) -> ndarray:
    u_axes = _unify_tensordot_axis((a.ndim, b.ndim), axes)
    assert a.dtype == b.dtype
    if _tensordot_convforward_check(a, b, u_axes):
        return _tensordot_convforward(a, b)
    if _tensordot_convbackwardinput_check(a, b, u_axes):
        return _tensordot_convbackwardinput(a, b)
    if _tensordot_convbackwardweight_check(a, b, u_axes):
        return _tensordot_convbackwardweight(a, b)

    return _tensordot_generic(a, b, u_axes)
