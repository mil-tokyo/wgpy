import re
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from wgpy.construct import asarray
from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
from wgpy_backends.webgpu.ndarray import ndarray

class OpWithType(NamedTuple):
    routine: str
    in_dtypes: List[np.dtype]
    out_dtype: np.dtype # support only one output

class ufunc:
    name: str
    types: List[str]
    ops: List[OpWithType]
    nin: int
    nout: int
    nargs: int
    _kernels: Dict[Tuple[str, str], ElementwiseKernel]

    def __init__(self, name: str, nin: int, nout: int, ops: List[Tuple[str, str]]) -> None:
        self.name = name
        self.nin = nin
        # nout != 1 is very rare and requires much differ implementation in webgl
        assert nout == 1
        self.nout = nout
        self.nargs = nin + nout
        types = []
        parsed_ops = []
        for op_dtype, routine in ops:
            types.append(op_dtype)
            in_dtypes = [np.dtype(t) for t in op_dtype[:-3]] # 'ii->i'
            out_dtype = np.dtype(op_dtype[-1])
            parsed_ops.append(OpWithType(routine=routine, in_dtypes=in_dtypes, out_dtype=out_dtype))
        self.types = types
        self.ops = parsed_ops
        self._kernels = {}

    @property
    def ntypes(self):
        return len(self.types)

    def reduce(self, array, axis=0, dtype=None, out=None, keepdims=False, initial=None, where=True):
        raise NotImplementedError
    
    def accumulate(self, array, axis=0, dtype=None, out=None):
        raise NotImplementedError
    
    def reduceat(self, array, indices, axis=0, dtype=None, out=None):
        raise NotImplementedError

    def outer(self, A, B, /, **kwargs):
        raise NotImplementedError
    
    def at(self, a, indices, b=None, /):
        raise NotImplementedError
    
    def __call__(self, *args: List[ndarray], out=None, where=None, axes=None, axis=None, keepdims=False, casting='same_kind', order='K', dtype=None, subok=True, signature=None, extobj=None) -> ndarray:
        assert where is None
        assert axes is None
        assert axis is None
        assert keepdims is False
        assert casting == 'same_kind'
        assert order == 'K'
        assert signature is None
        # assert dtype is None
        if dtype is not None:
            dtype = np.dtype(dtype)

        in_arrays = [asarray(array) for array in args[:self.nin]]
        assert len(in_arrays) == self.nin
        out_array = None
        if len(args) == self.nargs:
            out_array = args[-1]
        if out is not None:
            assert out_array is None
            out_array = out
        
        # Find op that matches the type
        # No automatic cast is performed
        matched_op = None
        def find_matched_op():
            for op in self.ops:
                ok = True
                for in_array, in_dtype in zip(in_arrays, op.in_dtypes):
                    if in_array.dtype != in_dtype:
                        ok = False
                        break
                if out_array is not None:
                    if out_array.dtype != op.out_dtype:
                        ok = False
                if dtype is not None:
                    if dtype != op.out_dtype:
                        ok = False
                if ok:
                    return op
            return None

        matched_op = find_matched_op()
        if matched_op is None:
            # adhoc cast
            # TODO: cast rule
            in_types = [array.dtype for array in in_arrays]
            dst_dtype = np.dtype(np.float32)
            if dst_dtype in in_types:
                in_arrays = [array.astype(dst_dtype) if array.dtype != dst_dtype else array for array in in_arrays]
                matched_op = find_matched_op()


        assert matched_op is not None, f"ufunc: type assignment failed for input types={[ary.dtype for ary in in_arrays]}"
        
        if out_array is None:
            # broadcasting
            target_shapes = [array.shape for array in in_arrays]
            result_shape = np.broadcast_shapes(*target_shapes)
            out_array = ndarray(result_shape, matched_op.out_dtype)
        else:
            assert isinstance(out_array, ndarray)
        
        # Cannot assign the same texture as both input and output or as multiple inputs in WebGL
        # In such cases, copy the input side
        bound_buffers = {id(out_array.buffer)}
        for i in range(len(in_arrays)):
            if id(in_arrays[i].buffer) in bound_buffers:
                in_arrays[i] = in_arrays[i].copy()
            bound_buffers.add(id(in_arrays[i].buffer))

        in_params = ','.join([f"{'TUVWXYZ'[i]} in{i}" for i in range(self.nin)])
        out_params = f"{'TUVWXYZ'[self.nin]} out0"
        kernel_key = (in_params, out_params, matched_op.out_dtype, tuple(matched_op.in_dtypes))
        if kernel_key in self._kernels:
            kernel = self._kernels[kernel_key]
        else:
            kernel = ElementwiseKernel(in_params=in_params, out_params=out_params, operation=matched_op.routine, name=self.name)
            self._kernels[kernel_key] = kernel
        kernel(*in_arrays, out_array)
        return out_array

def create_ufunc(name: str, ops, routine) -> ufunc:
    normalized_ops = []
    nout = 1
    nin = None
    for op in ops:
        if isinstance(op, tuple):
            op_type, sp_routine = op
        else:
            op_type = op
            sp_routine = routine
        assert re.match('^[?Bif]+->[?Bif]$', op_type) is not None
        nin_op = len(op_type) - 3
        if nin is None:
            nin = nin_op
        else:
            assert nin == nin_op, "all ops must have same number of inputs"
        normalized_ops.append((op_type, sp_routine))
    assert nin is not None
    return ufunc(name=name, nin=nin, nout=nout, ops=normalized_ops)
