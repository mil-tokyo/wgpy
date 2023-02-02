from wgpy import *
import cupy.cuda
import cupyx
from cupy_backends.runtime import mock_elementwise_kernel, mock_reduction_kernel

_default_memory_pool = None
def get_default_memory_pool():
    global _default_memory_pool
    if _default_memory_pool is None:
        _default_memory_pool = cupy.cuda.MemoryPool()
    return _default_memory_pool

_default_pinned_memory_pool = None
def get_default_pinned_memory_pool():
    global _default_pinned_memory_pool
    if _default_pinned_memory_pool is None:
        _default_pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
    return _default_pinned_memory_pool

class ElementwiseKernel:
    def __init__(self, in_params, out_params, operation, name='kernel', reduce_dims=True, preamble='', no_return=False, return_tuple=False, loop_prep='', after_loop=''):
        self.in_params = in_params
        self.out_params = out_params
        self.operation = operation
        self.name = name
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.no_return = no_return
        self.return_tuple = return_tuple
        self.loop_prep = loop_prep
        self.after_loop = after_loop

    def __call__(self, *args, size=None, block_size=None):
        return mock_elementwise_kernel(self, args, size=size, block_size=block_size)

class ReductionKernel:
    def __init__(self, in_params, out_params, map_expr, reduce_expr, post_map_expr, identity, name='reduce_kernel', reduce_type=None, reduce_dims=True, preamble='', options=()):
        self.in_params = in_params
        self.out_params = out_params
        self.map_expr = map_expr
        self.reduce_expr = reduce_expr
        self.post_map_expr = post_map_expr
        self.identity = identity
        self.name = name
        self.reduce_type = reduce_type
        self.reduce_dims = reduce_dims
        self.preamble = preamble
        self.options = options
    
    def __call__(self, *args, out=None, axis=None, keepdims=False, stream=None):
        return mock_reduction_kernel(self, args, out=out, axis=axis, keepdims=keepdims, stream=stream)
