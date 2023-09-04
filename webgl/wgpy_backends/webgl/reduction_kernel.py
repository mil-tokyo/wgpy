from typing import List, Optional, Tuple, Union
import numpy as np
from wgpy.construct import asarray
from wgpy.common.shape_util import calculate_c_contiguous_strides
from wgpy_backends.webgl.texture import WebGL2RenderingContext, WebGLArrayTextureShape
from wgpy_backends.webgl.ndarray import ndarray
from wgpy_backends.webgl.shader_util import header, native_scalar_type_for_type, native_pixel_type_for_internal_format
from wgpy_backends.webgl.kernel_common import GenericResolveResult, make_input_reduction_uniform, make_input_uniform, make_output_uniform, parse_in_params, parse_out_params, parse_uniforms, resolve_generic_type, InParam, OutParam, UniformDefinition
from wgpy_backends.webgl.platform import get_platform

added_kernels = set()

def get_input_key(name, out_name, ndim, elementwise, dtype, texture_shape: WebGLArrayTextureShape):
    """
    Obtain the key that is the branching factor for kernel generation
    """
    return (name, ndim, elementwise, dtype, texture_shape.dim, texture_shape.internal_format, texture_shape.format, texture_shape.type)

def make_output_key(name, ndim, dtype, texture_shape):
    """
    Obtain the key that is the branching factor for kernel generation
    """
    return (name, ndim, dtype, texture_shape.dim, texture_shape.internal_format, texture_shape.format, texture_shape.type)
    
def make_input_def(param: InParam, out_name, ndim, dtype, texture_shape: WebGLArrayTextureShape, axis: Tuple[int, ...]):
    elementwise = (not param.raw) and (not param.rawnd)
    # if rawnd
    #     create function name(index_0, index_1, ...) and used by user code
    # elif raw
    #     create function name(flat_index) and used by user code
    #     create function _name(index_0, index_1, ...) and used by name(flat_index)
    #     in this case, shape of array is needed as uniform.
    #     array is accessed as if it is c-contiguous. flat_index cannot be directly used to access texture, because array may be not c-contiguous. 
    # else
    #     create function _name(index_0, index_1, ...) and use it as name=_name(out_index_0, reduction_index_0, ...) in generated main function
    name = param.name
    uniform = ""
    sampler_type = {
        WebGL2RenderingContext.FLOAT: 'sampler2D',
        WebGL2RenderingContext.HALF_FLOAT: 'sampler2D',WebGL2RenderingContext.INT: 'isampler2D',
        WebGL2RenderingContext.UNSIGNED_BYTE: 'usampler2D',
    }[texture_shape.type]
    if texture_shape.dim == '2DArray':
        sampler_type += 'Array'
    uniform += f"uniform {sampler_type} _{name}_texture;\n"
    uniform += f"uniform int _{name}_offset;\n"
    if param.raw:
        for d in range(ndim):
            uniform += f"uniform int _{name}_shape_{d};\n"
    for d in range(ndim):
        uniform += f"uniform int _{name}_stride_{d};\n"
    native_texel_scalar_type = native_scalar_type_for_type[texture_shape.type]
    native_pixel_type = native_pixel_type_for_internal_format[texture_shape.internal_format]
    
    if texture_shape.dim == '2DArray':
        if texture_shape.elements_per_pixel == 1:
            func_def = f"""
{param.native_type_or_generic} {"_" if not param.rawnd else ""}{name}({",".join(f"int idx{d}" for d in range(ndim))})
{{
    int i = _{name}_offset{"".join(f" + _{name}_stride_{d} * idx{d}" for d in range(ndim))};
    ivec3 tsize = textureSize(_{name}_texture, 0);
    int y = i / tsize.x;
    int x = i - y * tsize.x;
    int z = y / tsize.y;
    y = y - z * tsize.y;
    {native_texel_scalar_type} v = texelFetch(_{name}_texture, ivec3(x, y, z), 0).r;
    return {param.native_type_or_generic}(v);
}}
"""
        else:
            func_def = f"""
{param.native_type_or_generic} {"_" if not param.rawnd else ""}{name}({",".join(f"int idx{d}" for d in range(ndim))})
{{
    int i = _{name}_offset{"".join(f" + _{name}_stride_{d} * idx{d}" for d in range(ndim))};
    ivec3 tsize = textureSize(_{name}_texture, 0);
    int e = i / 4;
    int c = i - e * 4;
    int y = e / tsize.x;
    int x = e - y * tsize.x;
    int z = y / tsize.y;
    y = y - z * tsize.y;
    {native_pixel_type} vvec = texelFetch(_{name}_texture, ivec3(x, y, z), 0);
    {native_texel_scalar_type} v;
    if (c == 1) {{ v = vvec.g; }}
    else if (c == 2) {{ v = vvec.b; }}
    else if (c == 3) {{ v = vvec.a; }}
    else {{ v = vvec.r; }}
    return {param.native_type_or_generic}(v);
}}
"""
    else:
        if texture_shape.elements_per_pixel == 1:
            func_def = f"""
{param.native_type_or_generic} {"_" if not param.rawnd else ""}{name}({",".join(f"int idx{d}" for d in range(ndim))})
{{
    int i = _{name}_offset{"".join(f" + _{name}_stride_{d} * idx{d}" for d in range(ndim))};
    ivec2 tsize = textureSize(_{name}_texture, 0);
    int y = i / tsize.x;
    int x = i - y * tsize.x;
    {native_texel_scalar_type} v = texelFetch(_{name}_texture, ivec2(x, y), 0).r;
    return {param.native_type_or_generic}(v);
}}
"""
        else:
            func_def = f"""
{param.native_type_or_generic} {"_" if not param.rawnd else ""}{name}({",".join(f"int idx{d}" for d in range(ndim))})
{{
    int i = _{name}_offset{"".join(f" + _{name}_stride_{d} * idx{d}" for d in range(ndim))};
    ivec2 tsize = textureSize(_{name}_texture, 0);
    int e = i / 4;
    int c = i - e * 4;
    int y = e / tsize.x;
    int x = e - y * tsize.x;
    {native_pixel_type} vvec = texelFetch(_{name}_texture, ivec2(x, y), 0);
    {native_texel_scalar_type} v;
    if (c == 1) {{ v = vvec.g; }}
    else if (c == 2) {{ v = vvec.b; }}
    else if (c == 3) {{ v = vvec.a; }}
    else {{ v = vvec.r; }}
    return {param.native_type_or_generic}(v);
}}
"""

    if param.raw:
        func_def += f"""
{param.native_type_or_generic} {name}(int flat_idx) {{
"""
        func_def += f"int _{name}_t1 = flat_idx;\n"
        func_def += f"int _{name}_t2;\n"
        for d in range(ndim-1, 0, -1): # ndim-1, ndim-2, ..., 1
            func_def += f"""_{name}_t2 = _{name}_t1 / _{name}_shape_{d};
    int _{name}_{d} = _{name}_t1 - _{name}_t2 * _{name}_shape_{d};
    _{name}_t1 = _{name}_t2;
    """
        if ndim > 0:
            func_def += f"int _{name}_0 = _{name}_t1;\n"
        func_def += f"""
    return _{name}({','.join(f"_{name}_{dim}" for dim in range(ndim))});
}}
"""
    
    if elementwise:
        access_idxs = []
        keep_count = 0
        red_count = 0
        for d in range(ndim):
            if d in axis:
                access_idxs.append(f"_redi_{red_count}")
                red_count += 1
            else:
                access_idxs.append(f"_{out_name}_{keep_count}")
                keep_count += 1
        inner_head = f"""{param.native_type_or_generic} {name} = _{name}({",".join(access_idxs)});\n"""
    else:
        inner_head = ""
    return uniform, func_def, inner_head


def make_output_def(param: OutParam, ndim, dtype, texture_shape: WebGLArrayTextureShape):
    name = param.name
    uniform = ""
    for d in range(ndim):
        uniform += f"uniform int _{name}_shape_{d};\n"
    uniform += "uniform int _out_ind_size;\n" # cupy's _out_ind.size()
    uniform += f"uniform int _{name}_texture_w;\n"
    if texture_shape.dim == "2DArray":
        uniform += f"uniform int _{name}_texture_h;\n"
        uniform += f"uniform int _draw_depth;\n"
    native_texel_scalar_type = native_scalar_type_for_type[texture_shape.type]
    native_pixel_type = native_pixel_type_for_internal_format[texture_shape.internal_format]
    out_def = f"out {native_pixel_type} _out_color;\n"
    
    main_head = ""
    main_tail = ""
    loop_tail = ""
    main_tail = ""
    if texture_shape.elements_per_pixel == 4:
        main_head += f"{native_pixel_type} _{name}_vec;"
    loop_head = f"{param.native_type_or_generic} {name};\n"
    if texture_shape.dim == "2DArray":
        main_head += f"""
highp float _{name}_gfcx = gl_FragCoord.x;
highp float _{name}_gfcy = gl_FragCoord.y;
int i = int(_{name}_gfcx - 0.5) + _{name}_texture_w * (int(_{name}_gfcy - 0.5) + _draw_depth * _{name}_texture_h);
"""
    else:
        main_head += f"""
highp float _{name}_gfcx = gl_FragCoord.x;
highp float _{name}_gfcy = gl_FragCoord.y;
int i = int(_{name}_gfcx - 0.5) + _{name}_texture_w * int(_{name}_gfcy - 0.5);
"""
    if texture_shape.elements_per_pixel == 4:
        main_head += f"i *= 4;"
    loop_head += f"int _{name}_t1 = i;\n"
    loop_head += f"int _{name}_t2;\n"
    for d in range(ndim-1, 0, -1): # ndim-1, ndim-2, ..., 1
        loop_head += f"""_{name}_t2 = _{name}_t1 / _{name}_shape_{d};
int _{name}_{d} = _{name}_t1 - _{name}_t2 * _{name}_shape_{d};
_{name}_t1 = _{name}_t2;
"""
    if ndim > 0:
        loop_head += f"int _{name}_0 = _{name}_t1;\n"
        loop_head += f"""if (_{name}_0 >= _{name}_shape_{0}) {{ {"return" if texture_shape.elements_per_pixel == 1 else "break"}; }}\n"""
    if texture_shape.elements_per_pixel == 4:
        loop_tail += f"""
if (_rgba_loop == 1) {{ _{name}_vec.g = {native_texel_scalar_type}({name}); }}
else if (_rgba_loop == 2) {{ _{name}_vec.b = {native_texel_scalar_type}({name}); }}
else if (_rgba_loop == 3) {{ _{name}_vec.a = {native_texel_scalar_type}({name}); }}
else {{ _{name}_vec.r = {native_texel_scalar_type}({name}); }}
i++;
        """
        main_tail += f"_out_color = _{name}_vec;"
    else:
        main_tail += f"_out_color = {native_texel_scalar_type}({name});"
    return uniform, out_def, loop_head, loop_tail, main_head, main_tail

def make_uniform_def(parsed_uniforms: List[UniformDefinition]):
    uniform = ""
    for uniform_def in parsed_uniforms:
        uniform += f"uniform {uniform_def.native_type} {uniform_def.name};\n"
    return uniform

def make_reduction_loop(axis: Tuple[int, ...], input_shape: Tuple[int, ...], out_texture_shape: WebGLArrayTextureShape, reduce_type: Optional[str], identity: str):
    if reduce_type is None:
        reduce_type = native_scalar_type_for_type[out_texture_shape.type]
    loop_head = f"{reduce_type} a = {identity}, b;\n"
    reduction_define = ""
    reduction_loop_open = ""
    reduction_loop_close = ""
    for i, a in enumerate(axis):
        reduction_define += f"#define _redi_shape_{i} {input_shape[a]}\n"
        reduction_loop_open += f"for (int _redi_{i} = 0; _redi_{i} < _redi_shape_{i}; _redi_{i}++) {{\n"
        reduction_loop_close += "}\n"
    return reduction_define, reduction_loop_open, reduction_loop_close, loop_head

class ReductionKernel:
    _next_idx = 1

    def __init__(self, in_params: str, out_params: str, map_expr: str, reduce_expr: str, post_map_expr: str, identity: str, name: str, reduce_type: Optional[str]=None, uniforms: str='', preamble: str='', no_return: bool=False, return_tuple: bool=False) -> None:
        self.parsed_in_params = parse_in_params(in_params)
        self.parsed_out_param = parse_out_params(out_params)
        self.parsed_uniforms = parse_uniforms(uniforms)
        self.nin = len(self.parsed_in_params)
        self.nout = 1
        self.in_params = in_params
        self.out_params = out_params
        self.uniforms = uniforms
        self.map_expr = map_expr
        self.reduce_expr = reduce_expr
        self.post_map_expr = post_map_expr
        self.identity = identity
        self.name = name
        self.reduce_type = reduce_type
        self.preamble = preamble
        self.no_return = no_return
        self.return_tuple = return_tuple

        self.kernel_name_prefix = f"reduce_{ReductionKernel._next_idx}_{name}_"
        ReductionKernel._next_idx += 1
        self.kernel_keys = {}

    def _generate_kernel_source(self, in_array_impls: List[ndarray], out_array_impl: ndarray, generic_resolve_result: GenericResolveResult, axis: Tuple[int, ...], input_shape: Tuple[int, ...]):
        uniform_all = ""
        uniform_all += "uniform int _in_ind_size;\n" # cupy's _in_ind.size()
        out_def_all = ""
        func_def_all = ""
        loop_head_all = ""
        loop_tail_all = ""
        main_head_all = ""
        main_tail_all = ""
        inner_head_all = ""
        uniform, out_def, loop_head, loop_tail, main_head, main_tail = make_output_def(self.parsed_out_param, out_array_impl.ndim, out_array_impl.dtype, out_array_impl.buffer.texture_shape)
        uniform_all += uniform
        out_def_all += out_def
        loop_head_all += loop_head
        loop_tail_all += loop_tail
        main_head_all += main_head
        main_tail_all += main_tail
        uniform = make_uniform_def(self.parsed_uniforms)
        reduction_define, reduction_loop_open, reduction_loop_close, loop_head = make_reduction_loop(axis, input_shape, out_array_impl.buffer.texture_shape, self.reduce_type, self.identity)
        uniform_all += uniform
        loop_head_all += loop_head
        for k, ary in zip(self.parsed_in_params, in_array_impls):
            uniform, func_def, inner_head = make_input_def(k, self.parsed_out_param.name, ary.ndim, ary.dtype, ary.buffer.texture_shape, axis)
            uniform_all += uniform
            func_def_all += func_def
            inner_head_all += inner_head

        if out_array_impl.buffer.texture_shape.elements_per_pixel == 1:
            source = f"""{header}
{self.preamble}
{reduction_define}
{generic_resolve_result.define_statements}
{uniform_all}
{out_def_all}
{func_def_all}
void main() {{
{main_head_all}
{loop_head_all}
{reduction_loop_open}
{inner_head_all}
b = ({self.map_expr});
a = ({self.reduce_expr});
{reduction_loop_close}
{self.post_map_expr};
{loop_tail_all}
{main_tail_all}
}}
"""
        else:
            # independently compute RGBA values by loop
            source = f"""{header}
{self.preamble}
{reduction_define}
{generic_resolve_result.define_statements}
{uniform_all}
{out_def_all}
{func_def_all}
void main() {{
{main_head_all}
for (int _rgba_loop = 0; _rgba_loop < 4; _rgba_loop++) {{
{loop_head_all}
{reduction_loop_open}
{inner_head_all}
b = ({self.map_expr});
a = ({self.reduce_expr});
{reduction_loop_close}
{self.post_map_expr};
{loop_tail_all}
}}
{main_tail_all}
}}
"""
        return source

    def _normalize_axis(self, axis: Optional[Union[int, Tuple[int, ...]]], ndim_input: int) -> List[int]:
        if axis is None:
            n_axis = list(range(ndim_input))
        elif isinstance(axis, int):
            if axis < 0:
                axis = axis + ndim_input
            n_axis = [axis]
        else:
            n_axis = []
            for a in axis:
                if a < 0:
                    a = a + ndim_input
                n_axis.append(a)
            n_axis.sort()
        return n_axis

    def __call__(self, *arrays, axis: Optional[Union[int, Tuple[int, ...]]]=None, keepdims: bool=False, uniforms: Optional[dict]=None) -> ndarray:
        assert len(arrays) == self.nin or len(arrays) == self.nin + self.nout

        in_arrays = [asarray(array) for array in arrays[:self.nin]]
        out_array = None
        if len(arrays) == self.nin + self.nout:
            out_array = arrays[self.nin] # maybe None
        
        # broadcasting
        target_shapes = []
        for i, array in enumerate(arrays):
            if array is None:
                continue
            if i < len(self.parsed_in_params):
                pip = self.parsed_in_params[i]
                if pip.raw or pip.rawnd:
                    continue
            target_shapes.append(array.shape)
        input_shape = np.broadcast_shapes(*target_shapes)

        in_array_impls = [] # type: List[ndarray]
        for i, array in enumerate(in_arrays):
            pip = self.parsed_in_params[i]
            if pip.raw or pip.rawnd:
                in_array_impls.append(array)
            else:
                in_array_impls.append(array.broadcast_to(input_shape))

        # reduce by axis
        n_axis = self._normalize_axis(axis, len(input_shape))
        axis_keys = [] # part of kernel_key
        result_shape_squeeze = []
        result_shape_keepdims = []
        for dim in range(len(input_shape)):
            if dim in n_axis:
                axis_keys.append((dim, input_shape[dim]))
                result_shape_keepdims.append(1)
            else:
                result_shape_squeeze.append(input_shape[dim])
                result_shape_keepdims.append(input_shape[dim])
        result_shape_squeeze = tuple(result_shape_squeeze)
        result_shape_keepdims = tuple(result_shape_keepdims)
        result_shape = result_shape_keepdims if keepdims else result_shape_squeeze

        # note: type casting is not performed
        generic_resolve_result = resolve_generic_type(in_arrays, out_array, self.parsed_in_params, self.parsed_out_param)
        if out_array is None:
            out_array = ndarray(result_shape, generic_resolve_result.out_dtype)
        else:
            assert out_array.shape == result_shape
            assert out_array.dtype == generic_resolve_result.out_dtype
            if not out_array.flags.c_contiguous_full:
                raise NotImplementedError
        out_array_impl_squeeze = out_array.get_view(result_shape_squeeze, out_array.dtype, calculate_c_contiguous_strides(result_shape_squeeze, out_array.itemsize), out_array.offset)
        
        # even if same instance, different source code is generated for dtype, ndim etc.
        # assigning unique key among same application.
        kernel_key = (tuple(get_input_key(k.name, self.parsed_out_param.name, ary.ndim, True, ary.dtype, ary.buffer.texture_shape) for k, ary in zip(self.parsed_in_params, in_array_impls)), make_output_key(self.parsed_out_param.name, out_array_impl_squeeze.ndim, out_array_impl_squeeze.dtype, out_array_impl_squeeze.buffer.texture_shape), tuple(axis_keys))
        kernel_name = self.kernel_keys.get(kernel_key, None)

        if kernel_name is None:
            kernel_name = self.kernel_name_prefix + str(len(self.kernel_keys))
            self.kernel_keys[kernel_key] = kernel_name
        if kernel_name not in added_kernels:
            source = self._generate_kernel_source(in_array_impls, out_array_impl_squeeze, generic_resolve_result, n_axis, input_shape)
            get_platform().addKernel(kernel_name, {'source': source})
            added_kernels.add(kernel_name)
        all_uniforms = []
        for k, array in zip(self.parsed_in_params, in_array_impls):
            all_uniforms.extend(make_input_uniform(k, array))
        all_uniforms.extend(make_input_reduction_uniform(input_shape))
        all_uniforms.extend(make_output_uniform(self.parsed_out_param, out_array_impl_squeeze, True))
        if uniforms is not None:
            for uniform_def in self.parsed_uniforms:
                all_uniforms.append({'type': uniform_def.native_type, 'name': uniform_def.name, 'value': float(uniforms[uniform_def.name])})
        else:
            assert len(self.parsed_uniforms) == 0

        get_platform().runKernel({'name': kernel_name, 'inputs': [{'name': f'_{k.name}_texture', 'id': array.buffer.buffer_id} for k, array in zip(self.parsed_in_params, in_arrays)], 'output': out_array_impl_squeeze.buffer.buffer_id, 'uniforms': all_uniforms})
        if self.no_return:
            return None
        if self.return_tuple:
            return out_array,
        return out_array
