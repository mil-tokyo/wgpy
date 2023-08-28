import math
from typing import List, Optional, Tuple, Union
from wgpy_backends.webgpu import common_reduction
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
        self.reduction = common_reduction
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

    def _unify_tensordot_axis(self, ndims: Tuple[int, int], axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]]) -> Tuple[List[int], List[int]]:
        if isinstance(axes, int):
            # axes=2: ([ndims[0]-1,ndims[0]-2], [0, 1])
            return [[ndims[0]-i-1 for i in range(axes)], [i for i in range(axes)]]
        if isinstance(axes[0], int):
            assert isinstance(axes[1], int)
            return ([axes[0]], [axes[1]])
        assert len(axes[0]) == len(axes[1])
        return axes

    def tensordot(self, a: ndarray, b: ndarray, axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]]=2) -> ndarray:
        # TODO: optimize for convolution
        # convolution forward: tensordot(col(ndim=6), W(ndim=4), axes=((1,2,3),(1,2,3)))
        # convolution backward col: tensordot(W(ndim=4), x(ndim=4), axes=(0, 1))
        # convolution backward weight: tensordot(gy(ndim=4), col(ndim=6), axes=((0,2,3),(0,4,5)))
        u_axes = self._unify_tensordot_axis((a.ndim, b.ndim), axes)
        assert a.dtype == b.dtype

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

        kernel_key = ('tensordot', a.ndim, b.ndim, tuple(u_axes[0]), tuple(u_axes[1]), tuple(ip_shape))
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
                    a_keys.append(f'ip{ip_idx}')
                except ValueError:
                    a_keys.append(f'_out0_{out_count}')
                    out_count += 1
            for dim in range(b.ndim):
                try:
                    ip_idx = u_axes[1].index(dim)
                    b_keys.append(f'ip{ip_idx}')
                except ValueError:
                    b_keys.append(f'_out0_{out_count}')
                    out_count += 1
            lf = '\n'
            source = f'''
{lf.join(f"const IP{dim}: i32 = {ip_shape[dim]};" for dim in range(len(ip_shape)))}

out0 = T(0);

{lf.join(f"for (var ip{dim}: i32 = 0; ip{dim} < IP{dim}; ip{dim}++){{" for dim in range(len(ip_shape)))}
out0 += a({','.join(a_keys)}) * b({','.join(b_keys)});
{lf.join(f"}}" for _ in range(len(ip_shape)))}

'''
            from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
            kernel = ElementwiseKernel(
                in_params="rawnd T a, rawnd T b",
                out_params="T out0",
                operation=source,
                name="tensordot"
            )
            elementwise_kernels[kernel_key] = kernel
        from wgpy.construct import empty
        out = empty(tuple(result_shape), dtype=a.dtype)
        kernel(a, b, out)
        return out
