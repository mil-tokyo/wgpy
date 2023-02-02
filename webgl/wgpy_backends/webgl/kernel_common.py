# shared routines for elementwise_kernel and reduction_kernel

import re
from typing import List, NamedTuple, Optional, Tuple
import numpy as np
from wgpy_backends.webgl.ndarray import ndarray
from wgpy_backends.webgl.shader_util import native_scalar_type_for_dtype, native_scalar_type_to_default_dtype

class InParam(NamedTuple):
    name: str
    native_type_or_generic: str # native type or generic
    generic: bool
    raw: bool
    rawnd: bool

class OutParam(NamedTuple):
    name: str
    native_type_or_generic: str # native type or generic
    generic: bool

class UniformDefinition(NamedTuple):
    name: str
    native_type: str # generic is not supported

def parse_in_params(in_params: str) -> List[InParam]:
    if re.match('^\\s*$', in_params):
        return []
    params = []
    for param in in_params.split(','):
        m = re.match("^\\s*(?:(raw|rawnd)\\s+)?(float|int|uint|[A-Z])\\s+([a-zA-Z][a-zA-Z0-9_]*)\\s*$", param)
        assert m is not None, f"syntax error in in_params: {in_params}"
        rawnd, type, name = m.groups()
        params.append(InParam(name=name, native_type_or_generic=type, generic=len(type)==1, raw=rawnd=='raw', rawnd=rawnd=='rawnd'))
    return params

def parse_out_params(out_params: str) -> OutParam:
    # webgl only support one output param, no "rawnd" accepted
    m = re.match("^\\s*(float|int|uint|[A-Z])\\s+([a-zA-Z][a-zA-Z0-9_]*)\\s*$", out_params)
    assert m is not None, f"syntax error in out_params: {out_params}"
    type, name = m.groups()
    return OutParam(name=name, native_type_or_generic=type, generic=len(type)==1)

def parse_uniforms(uniforms: str) -> List[UniformDefinition]:
    if re.match('^\\s*$', uniforms):
        return []
    params = []
    for param in uniforms.split(','):
        m = re.match("^\\s*(float|int|uint)\\s+([a-zA-Z][a-zA-Z0-9_]*)\\s*$", param)
        assert m is not None, f"syntax error in uniforms: {uniforms}"
        type, name = m.groups()
        params.append(UniformDefinition(name=name, native_type=type))
    return params

class GenericResolveResult(NamedTuple):
    define_statements: str
    out_dtype: np.dtype

def resolve_generic_type(in_array_impls: List[ndarray], explicit_out_array_impl: Optional[ndarray], parsed_in_params: List[InParam], parsed_out_param: OutParam) -> GenericResolveResult:
    # Is the fixed type appropriate for the actual data?
    # Assign GLSL type to generic type and generate define statement
    # Assign if output type is undefined
    generic_assignments = {}
    for k, array_impl in zip(parsed_in_params, in_array_impls):
        #array_native_type = native_scalar_type_for_type[array.buffer.texture_shape.type]
        array_native_type = native_scalar_type_for_dtype[array_impl.dtype]
        array_dtype = array_impl.dtype
        if k.generic:
            already_assigned_type = generic_assignments.get(k.native_type_or_generic)
            if already_assigned_type is not None:
                assert already_assigned_type == (array_dtype, array_native_type), f"ElementwiseKernel/ReductionKernel: generic resolve failed: already_assigned_type={already_assigned_type} != actual_type={(array_dtype, array_native_type)}"
            else:
                generic_assignments[k.native_type_or_generic] = (array_dtype, array_native_type)
        else:
            assert k.native_type_or_generic == array_native_type
    
    k = parsed_out_param
    if explicit_out_array_impl is not None:
        out_dtype = explicit_out_array_impl.dtype
        #array_native_type = native_scalar_type_for_type[explicit_out_array.buffer.texture_shape.type]
        array_native_type = native_scalar_type_for_dtype[explicit_out_array_impl.dtype]
        array_dtype = explicit_out_array_impl.dtype
        if k.generic:
            already_assigned_type = generic_assignments.get(k.native_type_or_generic)
            if already_assigned_type is not None:
                assert already_assigned_type == (array_dtype, array_native_type), f"ElementwiseKernel/ReductionKernel: generic resolve failed: already_assigned_type={already_assigned_type} != actual_type={(array_dtype, array_native_type)}"
            else:
                generic_assignments[k.native_type_or_generic] = (array_dtype, array_native_type)
        else:
            assert k.native_type_or_generic == array_native_type
    else:
        # Choose output type
        if k.generic:
            already_assigned_type = generic_assignments.get(k.native_type_or_generic)
            if already_assigned_type is not None:
                out_dtype = already_assigned_type[0]
            else:
                raise ValueError
        else:
            out_dtype = native_scalar_type_to_default_dtype[k.native_type_or_generic]

    define_statements = ""
    for k, (_, array_native_type) in generic_assignments.items():
        define_statements += f"#define {k} {array_native_type}\n"
    return GenericResolveResult(define_statements=define_statements, out_dtype=out_dtype)

def make_input_uniform(param: InParam, webgl_array: ndarray):
    name = param.name
    uniforms = []
    uniforms.append({'type': 'int', 'name': f'_{name}_offset', 'value': webgl_array.offset // webgl_array.itemsize})
    if param.raw:
        for d in range(webgl_array.ndim):
            uniforms.append({'type': 'int', 'name': f'_{name}_shape_{d}', 'value': webgl_array.shape[d]})
    for d in range(webgl_array.ndim):
        uniforms.append({'type': 'int', 'name': f'_{name}_stride_{d}', 'value': webgl_array.strides[d] // webgl_array.itemsize})
    return uniforms

def make_input_reduction_uniform(input_shape: Tuple[int, ...]):
    uniforms = []
    uniforms.append({'type': 'int', 'name': f'_in_ind_size', 'value': int(np.prod(input_shape))})
    return uniforms

def make_output_uniform(param: OutParam, webgl_array: ndarray, reduction: bool):
    if not webgl_array.flags.c_contiguous_full:
        raise NotImplementedError('In-place update of array view is not yet implemented.')
    name = param.name
    uniforms = []
    uniforms.append({'type': 'int', 'name': f'_{name}_texture_w', 'value': webgl_array.buffer.texture_shape.width})
    if webgl_array.buffer.texture_shape.dim == '2DArray':
        uniforms.append({'type': 'int', 'name': f'_{name}_texture_h', 'value': webgl_array.buffer.texture_shape.height})

    for d in range(webgl_array.ndim):
        uniforms.append({'type': 'int', 'name': f'_{name}_shape_{d}', 'value': webgl_array.shape[d]})
    if reduction:
        # cupy's _out_ind.size()
        uniforms.append({'type': 'int', 'name': f'_out_ind_size', 'value': webgl_array.size})
    else:
        # elementwise
        # cupy's _ind.size()
        uniforms.append({'type': 'int', 'name': f'_ind_size', 'value': webgl_array.size})
    return uniforms
