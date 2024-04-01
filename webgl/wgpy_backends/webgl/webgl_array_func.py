from typing import List, Optional, Tuple, Union
import numpy as np
from wgpy_backends.webgl.texture import WebGL2RenderingContext
from wgpy_backends.webgl.platform import get_platform
from wgpy_backends.webgl.ndarray import ndarray
import wgpy_backends.webgl.common_ufunc as common_ufunc
import wgpy_backends.webgl.common_reduction as common_reduction

added_kernels = set()
elementwise_kernels = {}


class WebGLArrayFunc:
    _instance = None

    def __init__(self) -> None:
        self.ufunc = common_ufunc
        self.reduction = common_reduction

    @staticmethod
    def instance() -> "WebGLArrayFunc":
        if WebGLArrayFunc._instance is None:
            WebGLArrayFunc._instance = WebGLArrayFunc()
        return WebGLArrayFunc._instance

    def matmul(
        self, lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None
    ) -> ndarray:
        # 2D only
        # TODO: irregular texture
        m, k = lhs.shape
        k2, n = rhs.shape
        assert k == k2
        for in_array in [lhs, rhs]:
            assert in_array.buffer.texture_shape.dim == "2D"
            assert in_array.buffer.texture_shape.internal_format in (
                WebGL2RenderingContext.R32F,
                WebGL2RenderingContext.R16F,
            )
            assert in_array.buffer.texture_shape.format == WebGL2RenderingContext.RED
            assert in_array.buffer.texture_shape.type in (
                WebGL2RenderingContext.FLOAT,
                WebGL2RenderingContext.HALF_FLOAT,
            )
        if out is not None:
            assert out.flags.c_contiguous_full
            assert out.buffer.texture_shape.dim == "2D"
            assert out.buffer.texture_shape.internal_format in (
                WebGL2RenderingContext.R32F,
                WebGL2RenderingContext.R16F,
            )
            assert out.buffer.texture_shape.format == WebGL2RenderingContext.RED
            assert out.buffer.texture_shape.type in (
                WebGL2RenderingContext.FLOAT,
                WebGL2RenderingContext.HALF_FLOAT,
            )
            for in_array in [lhs, rhs]:
                assert out.buffer.buffer_id != in_array.buffer.buffer_id
            assert out.shape == (m, n)
        kernel_name = f"matmul_{m}_{n}_{k}"
        if kernel_name not in added_kernels:
            get_platform().addKernel(
                kernel_name,
                {
                    "source": f"""#version 300 es
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
    """
                },
            )
            added_kernels.add(kernel_name)
        if out is None:
            out = ndarray((m, n), lhs.dtype)
        get_platform().runKernel(
            {
                "name": kernel_name,
                "inputs": [
                    {"name": "tex_lhs", "id": lhs.buffer.buffer_id},
                    {"name": "tex_rhs", "id": rhs.buffer.buffer_id},
                ],
                "output": out.buffer.buffer_id,
                "uniforms": [
                    {
                        "name": "_ka_tex_output_texture_w",
                        "value": out.buffer.texture_shape.width,
                        "type": "int",
                    },
                    {
                        "name": "LHS_STRIDE_0",
                        "value": lhs.strides[0] // lhs.itemsize,
                        "type": "int",
                    },
                    {
                        "name": "LHS_STRIDE_1",
                        "value": lhs.strides[1] // lhs.itemsize,
                        "type": "int",
                    },
                    {
                        "name": "RHS_STRIDE_0",
                        "value": rhs.strides[0] // rhs.itemsize,
                        "type": "int",
                    },
                    {
                        "name": "RHS_STRIDE_1",
                        "value": rhs.strides[1] // rhs.itemsize,
                        "type": "int",
                    },
                ],
            }
        )
        return out

    def _unify_tensordot_axis(
        self,
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

    def tensordot(
        self,
        a: ndarray,
        b: ndarray,
        axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]] = 2,
    ) -> ndarray:
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
{lf.join(f"#define IP{dim} {ip_shape[dim]}" for dim in range(len(ip_shape)))}

out0 = T(0);

{lf.join(f"for (int ip{dim} = 0; ip{dim} < IP{dim}; ip{dim}++){{" for dim in range(len(ip_shape)))}
out0 += a({','.join(a_keys)}) * b({','.join(b_keys)});
{lf.join(f"}}" for _ in range(len(ip_shape)))}

"""
            from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

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
