from typing import List, Optional, Tuple, Union
from wgpy_backends.webgpu import common_reduction
from wgpy_backends.webgpu import common_ufunc
from wgpy_backends.webgpu.ndarray import ndarray
from wgpy_backends.webgpu.matmul import matmul_impl

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
        return matmul_impl(lhs, rhs, out)

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
