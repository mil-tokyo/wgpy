import numpy as np
from wgpy.construct import asarray, asnumpy


def crossent_fwd(
    reduction_kernel, args, out=None, axis=None, keepdims=False, stream=None
):
    t, log_y, n_channel, coeff, ignore_label = args
    t_flat = asnumpy(t).flatten()
    log_y_flat = asnumpy(log_y).flatten()
    reduced = 0.0
    for i in range(len(t_flat)):
        if t_flat[i] == ignore_label:
            reduced += 0.0
        else:
            reduced += log_y_flat[i * n_channel + t_flat[i]]
    reduced = reduced * -asnumpy(coeff)  # coeff is scalar
    return asarray(np.float32(reduced))


def mock_reduction_kernel(
    reduction_kernel, args, out=None, axis=None, keepdims=False, stream=None
):
    if reduction_kernel.name == "crossent_fwd":
        return crossent_fwd(
            reduction_kernel, args, out=out, axis=axis, keepdims=keepdims, stream=stream
        )
    else:
        msg = f"""Requested reduction kernel is not implemented in WebGL.
{repr({'in_params': reduction_kernel.in_params, 'out_params': reduction_kernel.out_params, 'map_expr': reduction_kernel.map_expr, 'reduce_expr': reduction_kernel.reduce_expr, 'post_map_expr': reduction_kernel.post_map_expr, 'identity': reduction_kernel.identity, 'name': reduction_kernel.name, 'reduce_type': reduction_kernel.reduce_type, 'reduce_dims': reduction_kernel.reduce_dims, 'preamble': reduction_kernel.preamble, 'options': reduction_kernel.options})}
arguments:
"""
        for arg in args:
            msg += f"class={arg.__class__}, shape={getattr(arg, 'shape') if hasattr(arg, 'shape') else 'scalar'}\n"

        raise NotImplementedError(msg)
