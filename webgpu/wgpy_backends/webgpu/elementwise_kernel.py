from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from wgpy_backends.webgpu.webgpu_buffer import WebGPUMetaBuffer, WebGPUMetaBufferItem, create_meta_buffer_from_structure
from wgpy.construct import asarray
from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape
from wgpy_backends.webgpu.ndarray import ndarray
from wgpy_backends.webgpu.shader_util import header
from wgpy_backends.webgpu.kernel_common import GenericResolveResult, make_input_uniform, make_output_uniform, InParam, OutParam, parse_in_params, parse_out_params, parse_uniforms, resolve_generic_type
from wgpy_backends.webgpu.platform import get_platform

_WORKGROUP_SIZE_X = 64
_N_WORKGROUPS_X = 64

added_kernels = set()

def get_input_key(name, out_name, ndim, elementwise, dtype, texture_shape: WebGPUArrayTextureShape):
    """
    Obtain the key that is the branching factor for kernel generation
    """
    return (name, ndim, elementwise, dtype, texture_shape.logical_dtype)

def make_output_key(name, ndim, dtype, texture_shape: WebGPUArrayTextureShape):
    """
    Obtain the key that is the branching factor for kernel generation
    """
    return (name, ndim, dtype, texture_shape.logical_dtype)

def make_input_def(param: InParam, out_name, ndim, dtype, texture_shape: WebGPUArrayTextureShape, binding_index: int):
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
    meta_defs = [] # type: List[WebGPUMetaBufferItem]

    meta_defs.append(WebGPUMetaBufferItem(f"_{name}_offset", 'i32'))
    if param.raw:
        for d in range(ndim):
            meta_defs.append(WebGPUMetaBufferItem(f"_{name}_shape_{d}", 'i32'))
    for d in range(ndim):
        meta_defs.append(WebGPUMetaBufferItem(f"_{name}_stride_{d}", 'i32'))

    func_def = f"""
fn {"_" if not param.rawnd else ""}{name}({",".join(f"idx{d}: i32" for d in range(ndim))}) -> {param.native_type_or_generic}
{{
var i: i32 = cmeta._{name}_offset{"".join(f" + cmeta._{name}_stride_{d} * idx{d}" for d in range(ndim))};
var v: {texture_shape.storage_dtype} = _{name}_storage[i];
return {param.native_type_or_generic}(v);
}}
"""

    if param.raw:
        func_def += f"""
fn {name}(flat_idx: i32) -> {param.native_type_or_generic} {{
"""
        func_def += f"var _{name}_t1: i32 = flat_idx;\n"
        func_def += f"var _{name}_t2: i32;\n"
        for d in range(ndim-1, 0, -1): # ndim-1, ndim-2, ..., 1
            func_def += f"""_{name}_t2 = _{name}_t1 / cmeta._{name}_shape_{d};
    var _{name}_{d}: i32 = _{name}_t1 - _{name}_t2 * cmeta._{name}_shape_{d};
    _{name}_t1 = _{name}_t2;
    """
        if ndim > 0:
            func_def += f"var _{name}_0: i32 = _{name}_t1;\n"
        func_def += f"""
    return _{name}({','.join(f"_{name}_{dim}" for dim in range(ndim))});
}}
"""
    
    if elementwise:
        loop_head = f"""var {name}: {param.native_type_or_generic} = _{name}({",".join(f"_{out_name}_{d}" for d in range(ndim))});\n"""
    else:
        loop_head = ""
    variable_binding_source = f"""
@group(0) @binding({binding_index})
var<storage,read> _{name}_storage: array<{texture_shape.storage_dtype}>;
"""
    return meta_defs, func_def, loop_head, variable_binding_source


def make_output_def(param: OutParam, ndim, dtype, texture_shape: WebGPUArrayTextureShape, binding_index: int):
    name = param.name
    meta_defs = [] # type: List[WebGPUMetaBufferItem]

    for d in range(ndim):
        meta_defs.append(WebGPUMetaBufferItem(f"_{name}_shape_{d}", 'i32'))
    meta_defs.append(WebGPUMetaBufferItem(f"_ind_size", 'i32')) # cupy's _ind.size()

    main_head = ""
    main_tail = ""
    loop_tail = ""
    main_tail = ""
    loop_head = f"var {name}: {param.native_type_or_generic};\n"
    loop_head += f"var _{name}_t1: i32 = i;\n"
    loop_head += f"var _{name}_t2: i32;\n"
    for d in range(ndim-1, 0, -1): # ndim-1, ndim-2, ..., 1
        loop_head += f"""_{name}_t2 = _{name}_t1 / cmeta._{name}_shape_{d};
var _{name}_{d}: i32 = _{name}_t1 - _{name}_t2 * cmeta._{name}_shape_{d};
_{name}_t1 = _{name}_t2;
"""
    if ndim > 0:
        loop_head += f"var _{name}_0: i32 = _{name}_t1;\n"
        loop_head += f"""if (_{name}_0 > cmeta._{name}_shape_{0}) {{ break; }}\n"""
    loop_tail += f"""
        _{name}_storage[i] = {texture_shape.storage_dtype}({name});
        """
    variable_binding_source = f"""
@group(0) @binding({binding_index})
var<storage,read_write> _{name}_storage: array<{texture_shape.storage_dtype}>;
"""
    return meta_defs, loop_head, loop_tail, main_head, main_tail, variable_binding_source



class ElementwiseKernel:
    _next_idx = 1
    _meta_defs_for_kernel_key: Dict[str, List[WebGPUMetaBufferItem]]

    def __init__(self, in_params: str, out_params: str, operation: str, name: str, uniforms: Union[str, List[WebGPUMetaBufferItem]]='', preamble: str='', no_return: bool=False, return_tuple: bool=False) -> None:
        self.parsed_in_params = parse_in_params(in_params)
        self.parsed_out_param = parse_out_params(out_params)
        if isinstance(uniforms, str):
            self.meta_items = parse_uniforms(uniforms)
        elif isinstance(uniforms, list):
            self.meta_items = uniforms
        elif uniforms is None:
            self.meta_items = []
        else:
            raise TypeError(f"uniforms must be str or List[WebGPUMetaBufferItem], but got {type(uniforms)}")
        self.nin = len(self.parsed_in_params)
        self.nout = 1
        self.in_params = in_params
        self.out_params = out_params
        self.name = name
        self.operation = operation
        self.preamble = preamble
        self.no_return = no_return
        self.return_tuple = return_tuple

        self.kernel_name_prefix = f"elementwise_{ElementwiseKernel._next_idx}_{name}_"
        ElementwiseKernel._next_idx += 1
        self.kernel_keys = {}
        self._meta_defs_for_kernel_key = {}

    def _generate_kernel_source(self, in_array_impls: List[ndarray], out_array_impl: ndarray, generic_resolve_result: GenericResolveResult):
        func_def_all = ""
        loop_head_all = ""
        loop_tail_all = ""
        main_head_all = ""
        main_tail_all = ""
        meta_def_all = [] # type: List[WebGPUMetaBufferItem]
        variable_binding_source = """
@group(0) @binding(0)
var<storage,read> cmeta: CMeta;
"""
        binding_types = ['read-only-storage'] # type: List[str]
        next_binding_index = 1
        meta_defs, loop_head, loop_tail, main_head, main_tail, binding_source_part = make_output_def(self.parsed_out_param, out_array_impl.ndim, out_array_impl.dtype, out_array_impl.buffer.texture_shape, binding_index=next_binding_index)
        next_binding_index += 1
        binding_types.append('storage')
        meta_def_all.extend(meta_defs)
        loop_head_all += loop_head
        loop_tail_all += loop_tail
        main_head_all += main_head
        main_tail_all += main_tail
        variable_binding_source += binding_source_part
        for k, ary in zip(self.parsed_in_params, in_array_impls):
            meta_defs, func_def, loop_head, binding_source_part = make_input_def(k, self.parsed_out_param.name, ary.ndim, ary.dtype, ary.buffer.texture_shape, binding_index=next_binding_index)
            meta_def_all.extend(meta_defs)
            func_def_all += func_def
            loop_head_all += loop_head
            variable_binding_source += binding_source_part
            next_binding_index += 1
            binding_types.append('read-only-storage')
        meta_def_all.extend(self.meta_items)

        meta_def_source = f'struct CMeta {{{"".join([f"{meta.name}:{meta.native_type}," for meta in meta_def_all])}}}'

        source = f"""{header}
{self.preamble}
{generic_resolve_result.define_statements}
{meta_def_source}
{variable_binding_source}
{func_def_all}
@compute @workgroup_size({_WORKGROUP_SIZE_X},1,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {{
{main_head_all}
for (var i: i32 = i32(global_id.x);; i += {_WORKGROUP_SIZE_X * _N_WORKGROUPS_X}i) {{
{loop_head_all}
{self.operation};
{loop_tail_all}
}}
{main_tail_all}
}}
"""
        return source, meta_def_all, binding_types

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
            source, meta_defs, binding_types = self._generate_kernel_source(in_array_impls, out_array_impl, generic_resolve_result)
            get_platform().addKernel(kernel_name, {'source': source, 'bindingTypes': binding_types})
            added_kernels.add(kernel_name)
            self._meta_defs_for_kernel_key[kernel_name] = meta_defs
        all_uniforms = []
        for k, array in zip(self.parsed_in_params, in_array_impls):
            all_uniforms.extend(make_input_uniform(k, array))
        all_uniforms.extend(make_output_uniform(self.parsed_out_param, out_array_impl, False))
        if uniforms is not None:
            for uniform_def in self.meta_items:
                # TODO: missing uniform check
                all_uniforms.append({'type': uniform_def.native_type, 'name': uniform_def.name, 'value': uniforms[uniform_def.name]})
        else:
            assert len(self.meta_items) == 0

        meta_buffer = self._make_meta_buffer(self._meta_defs_for_kernel_key[kernel_name], all_uniforms)

        tensors = [meta_buffer.buffer_id, out_array_impl.buffer.buffer_id]
        for array in in_array_impls:
            tensors.append(array.buffer.buffer_id)
        get_platform().runKernel({'name': kernel_name,
                                  'tensors': tensors, 
                                  'workGroups': {'x': 64, 'y': 1, 'z': 1},})
        if self.no_return:
            return None
        if self.return_tuple:
            return out_array,
        return out_array

    def _make_meta_buffer(self, meta_defs: List[WebGPUMetaBufferItem], uniforms: List[dict]) -> WebGPUMetaBuffer:
        # uniform: {'type': 'i32', 'name': f'_ind_size', 'value': 123}
        values = []
        numpy_dtypes = []
        uniform_dict = {uniform['name']: uniform for uniform in uniforms}
        for meta_def in meta_defs:
            if meta_def.name in uniform_dict:
                values.append(uniform_dict[meta_def.name]['value'])
                numpy_dtypes.append(meta_def.numpy_dtype_str)
            else:
                raise KeyError(f"uniform {meta_def.name} is not found")
        return create_meta_buffer_from_structure(tuple(values), ",".join(numpy_dtypes))
