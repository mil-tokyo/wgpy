from typing import List, Optional, Tuple
import numpy as np
from wgpy.construct import asarray
from wgpy_backends.webgl.texture import WebGL2RenderingContext, WebGLArrayTextureShape
from wgpy_backends.webgl.ndarray import ndarray
from wgpy_backends.webgl.shader_util import header, native_scalar_type_for_type, native_pixel_type_for_internal_format
from wgpy_backends.webgl.kernel_common import GenericResolveResult, make_input_uniform, make_output_uniform, InParam, OutParam, UniformDefinition, parse_in_params, parse_out_params, parse_uniforms, resolve_generic_type
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

def make_input_def(param: InParam, out_name, ndim, dtype, texture_shape: WebGLArrayTextureShape):
    elementwise = (not param.raw) and (not param.rawnd)
    # if rawnd
    #     create function name(index_0, index_1, ...) and used by user code
    # elif raw
    #     create function name(flat_index) and used by user code
    #     create function _name(index_0, index_1, ...) and used by name(flat_index)
    #     in this case, shape of array is needed as uniform.
    #     array is accessed as if it is c-contiguous. flat_index cannot be directly used to access texture, because array may be not c-contiguous. 
    # else
    #     create function _name(index_0, index_1, ...) and use it as name=_name(out_index_0, out_index_1, ...) in generated main function
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
        loop_head = f"""{param.native_type_or_generic} {name} = _{name}({",".join(f"_{out_name}_{d}" for d in range(ndim))});\n"""
    else:
        loop_head = ""
    return uniform, func_def, loop_head


def make_output_def(param: OutParam, ndim, dtype, texture_shape: WebGLArrayTextureShape):
    name = param.name
    uniform = ""
    for d in range(ndim):
        uniform += f"uniform int _{name}_shape_{d};\n"
    uniform += "uniform int _ind_size;\n" # cupy's _ind.size()
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
        loop_head += f"""if (_{name}_0 > _{name}_shape_{0}) {{ {"return" if texture_shape.elements_per_pixel == 1 else "break"}; }}\n"""
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



class ElementwiseKernel:
    _next_idx = 1

    def __init__(self, in_params: str, out_params: str, operation: str, name: str, uniforms: str='', preamble: str='', no_return: bool=False, return_tuple: bool=False) -> None:
        self.parsed_in_params = parse_in_params(in_params)
        self.parsed_out_param = parse_out_params(out_params)
        self.parsed_uniforms = parse_uniforms(uniforms)
        self.nin = len(self.parsed_in_params)
        self.nout = 1
        self.in_params = in_params
        self.out_params = out_params
        self.uniforms = uniforms
        self.name = name
        self.operation = operation
        self.preamble = preamble
        self.no_return = no_return
        self.return_tuple = return_tuple

        self.kernel_name_prefix = f"elementwise_{ElementwiseKernel._next_idx}_{name}_"
        ElementwiseKernel._next_idx += 1
        self.kernel_keys = {}

    def _generate_kernel_source(self, in_array_impls: List[ndarray], out_array_impl: ndarray, generic_resolve_result: GenericResolveResult):
        uniform_all = ""
        out_def_all = ""
        func_def_all = ""
        loop_head_all = ""
        loop_tail_all = ""
        main_head_all = ""
        main_tail_all = ""
        uniform, out_def, loop_head, loop_tail, main_head, main_tail = make_output_def(self.parsed_out_param, out_array_impl.ndim, out_array_impl.dtype, out_array_impl.buffer.texture_shape)
        uniform_all += uniform
        out_def_all += out_def
        loop_head_all += loop_head
        loop_tail_all += loop_tail
        main_head_all += main_head
        main_tail_all += main_tail
        uniform = make_uniform_def(self.parsed_uniforms)
        uniform_all += uniform
        for k, ary in zip(self.parsed_in_params, in_array_impls):
            uniform, func_def, loop_head = make_input_def(k, self.parsed_out_param.name, ary.ndim, ary.dtype, ary.buffer.texture_shape)
            uniform_all += uniform
            func_def_all += func_def
            loop_head_all += loop_head

        if out_array_impl.buffer.texture_shape.elements_per_pixel == 1:
            source = f"""{header}
{self.preamble}
{generic_resolve_result.define_statements}
{uniform_all}
{out_def_all}
{func_def_all}
void main() {{
{main_head_all}
{loop_head_all}
{self.operation};
{loop_tail_all}
{main_tail_all}
}}
"""
        else:
            # independently compute RGBA values by loop
            source = f"""{header}
{self.preamble}
{generic_resolve_result.define_statements}
{uniform_all}
{out_def_all}
{func_def_all}
void main() {{
{main_head_all}
for (int _rgba_loop = 0; _rgba_loop < 4; _rgba_loop++) {{
{loop_head_all}
{self.operation};
{loop_tail_all}
}}
{main_tail_all}
}}
"""
        return source

    def __call__(self, *arrays, uniforms: Optional[dict]=None, size: Optional[Tuple[int]]=None) -> ndarray:
        assert size is None

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
        result_shape = np.broadcast_shapes(*target_shapes)

        in_array_impls = [] # type: List[ndarray]
        for i, array in enumerate(in_arrays):
            pip = self.parsed_in_params[i]
            if pip.raw or pip.rawnd:
                in_array_impls.append(array)
            else:
                in_array_impls.append(array.broadcast_to(result_shape))

        # note: type casting is not performed
        generic_resolve_result = resolve_generic_type(in_array_impls, out_array if out_array is not None else None, self.parsed_in_params, self.parsed_out_param)
        if out_array is None:
            out_array = ndarray(result_shape, generic_resolve_result.out_dtype)
        else:
            assert out_array.shape == result_shape
            assert out_array.dtype == generic_resolve_result.out_dtype
            if not out_array.flags.c_contiguous_full:
                raise NotImplementedError
        out_array_impl = out_array
        
        # even if same instance, different source code is generated for dtype, ndim etc.
        # assigning unique key among same application.
        kernel_key = (tuple(get_input_key(k.name, self.parsed_out_param.name, ary.ndim, True, ary.dtype, ary.buffer.texture_shape) for k, ary in zip(self.parsed_in_params, in_array_impls)), make_output_key(self.parsed_out_param.name, out_array_impl.ndim, out_array_impl.dtype, out_array_impl.buffer.texture_shape))
        kernel_name = self.kernel_keys.get(kernel_key, None)

        if kernel_name is None:
            kernel_name = self.kernel_name_prefix + str(len(self.kernel_keys))
            self.kernel_keys[kernel_key] = kernel_name
        if kernel_name not in added_kernels:
            source = self._generate_kernel_source(in_array_impls, out_array_impl, generic_resolve_result)
            get_platform().addKernel(kernel_name, {'source': source})
            added_kernels.add(kernel_name)
        all_uniforms = []
        for k, array in zip(self.parsed_in_params, in_array_impls):
            all_uniforms.extend(make_input_uniform(k, array))
        all_uniforms.extend(make_output_uniform(self.parsed_out_param, out_array_impl, False))
        if uniforms is not None:
            for uniform_def in self.parsed_uniforms:
                all_uniforms.append({'type': uniform_def.native_type, 'name': uniform_def.name, 'value': float(uniforms[uniform_def.name])})
        else:
            assert len(self.parsed_uniforms) == 0

        get_platform().runKernel({'name': kernel_name, 'inputs': [{'name': f'_{k.name}_texture', 'id': array.buffer.buffer_id} for k, array in zip(self.parsed_in_params, in_array_impls)], 'output': out_array_impl.buffer.buffer_id, 'uniforms': all_uniforms})
        if self.no_return:
            return None
        if self.return_tuple:
            return out_array,
        return out_array
