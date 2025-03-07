from functools import cache
import numpy as np
from wgpy.construct import asarray, asnumpy
from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel

_kernels = {}


def _register_kernel(name: str):
    def decorator(func):
        _kernels[name] = func

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


@_register_kernel("softmax_crossent_bwd")
def softmax_crossent_bwd(elementwise_kernel, args):
    y, t, coeff, n_channel, n_unit, ignore_label = args
    y_data = asnumpy(y)
    coeff_data = asnumpy(coeff)
    t_data_bc = np.broadcast_to(asnumpy(t), y_data.shape)
    i = np.arange(y_data.size).reshape(y_data.shape)
    c = i // n_unit % n_channel
    gx = np.where(
        t_data_bc == ignore_label, 0, coeff_data * (y_data - (c == t_data_bc))
    )
    return asarray(gx)


@_register_kernel("momentum_sgd")
def momentum_sgd(elementwise_kernel, args):
    grad, lr, momentum, param, v = args
    v[:] = momentum * v - lr * grad
    param += v
    return param, v


@cache
def relu_bwd_kernel():
    return ElementwiseKernel(
        in_params="f32 y, f32 gy",
        out_params="f32 gx",
        operation="if (y > 0.0) { gx = gy; } else { gx = 0.0; }",
        name="relu_bwd",
    )


@_register_kernel("relu_bwd")
def relu_bwd(elementwise_kernel, args):
    y, gy = args
    gx = relu_bwd_kernel()(y, gy)
    return gx


@cache
def max_pool_fwd_kernel_indexes():
    return ElementwiseKernel(
        in_params="raw T inx",
        out_params="i32 indexes",
        uniforms="i32 h, i32 w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let kw: i32 = cmeta.kw;
let kh: i32 = cmeta.kh;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let c: i32 = cmeta._indexes_shape_1;
let out_y: i32 = _indexes_2;
let out_x: i32 = _indexes_3;
let in_y_0: i32 = max(0, out_y * sy - ph);
let in_y_1: i32 = min(h, out_y * sy + kh - ph);
let in_x_0: i32 = max(0, out_x * sx - pw);
let in_x_1: i32 = min(w, out_x * sx + kw - pw);

var maxval: T = inx(((_indexes_0 * c + _indexes_1) * h + in_y_0) * w + in_x_0);
var argmax_y: i32 = in_y_0;
var argmax_x: i32 = in_x_0;
for (var yi: i32 = 0; yi < kh; yi++) {{
    var y: i32 = in_y_0 + yi;
    if (y >= in_y_1) {{
        break;
    }}
    for (var xi: i32 = 0; xi < kw; xi++) {{
        var x: i32 = in_x_0 + xi;
        if (x >= in_x_1) {{
            break;
        }}
        var v: T = inx(((_indexes_0 * c + _indexes_1) * h + y) * w + x);
        if (maxval < v) {{
            maxval = v;
            argmax_y = y;
            argmax_x = x;
        }}
    }}
}}

let argmax_ky: i32 = argmax_y + ph - out_y * sy;
let argmax_kx: i32 = argmax_x + pw - out_x * sx;
indexes = argmax_kx + kw * argmax_ky;
""",
        name=f"max_pool_fwd_indexes",
    )


@cache
def max_pool_fwd_kernel_maxval():
    return ElementwiseKernel(
        in_params="raw T inx, S indexes",
        out_params="T maxval",
        uniforms="i32 h, i32 w, i32 sy, i32 sx, i32 ph, i32 pw, i32 kw",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let kw: i32 = cmeta.kw;
let c: i32 = cmeta._maxval_shape_1;
let out_y: i32 = _maxval_2;
let out_x: i32 = _maxval_3;

var tmp: i32 = indexes;
var argmax_ky: i32 = tmp / kw;
var argmax_kx: i32 = tmp - kw * argmax_ky;
let argmax_y: i32 = argmax_ky + out_y * sy - ph;
let argmax_x: i32 = argmax_kx + out_x * sx - pw;

maxval = inx(((_maxval_0 * c + _maxval_1) * h + argmax_y) * w + argmax_x);
""",
        name="max_pool_fwd_maxval",
    )


@_register_kernel("max_pool_fwd")
def max_pool_fwd(elementwise_kernel, args):
    x, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, y, indexes = args
    max_pool_fwd_kernel_indexes()(
        x,
        indexes,
        uniforms={
            "h": h,
            "w": w,
            "kh": kh,
            "kw": kw,
            "sy": sy,
            "sx": sx,
            "ph": ph,
            "pw": pw,
        },
    )
    max_pool_fwd_kernel_maxval()(
        x,
        indexes,
        y,
        uniforms={
            "h": h,
            "w": w,
            #        'kh': kh,
            "kw": kw,
            "sy": sy,
            "sx": sx,
            "ph": ph,
            "pw": pw,
        },
    )
    return y, indexes


@cache
def max_pool_bwd_kernel():
    return ElementwiseKernel(
        in_params="raw T gy, raw i32 indexes",
        out_params="T gx",
        uniforms="i32 h, i32 w, i32 out_h, i32 out_w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let out_h: i32 = cmeta.out_h;
let out_w: i32 = cmeta.out_w;
let kh: i32 = cmeta.kh;
let kw: i32 = cmeta.kw;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let c: i32 = cmeta._gx_shape_1;
let y: i32 = _gx_2 + ph;
let x: i32 = _gx_3 + pw;
let out_y_0: i32 = max(0, (y - kh + sy) / sy);
let out_y_1: i32 = min(out_h, (y + sy) / sy);
let out_x_0: i32 = max(0, (x - kw + sx) / sx);
let out_x_1: i32 = min(out_w, (x + sx) / sx);

gx = T(0);
for (var yi: i32 = 0; yi < kh; yi++) {{
    var out_y: i32 = out_y_0 + yi;
    if (out_y >= out_y_1) {{
        break;
    }}
    var ky: i32 = y - out_y * sy;
    for (var xi: i32 = 0; xi < kw; xi++) {{
        var out_x: i32 = out_x_0 + xi;
        if (out_x >= out_x_1) {{
            break;
        }}
        var kx: i32 = x - out_x * sx;
        var offset: i32 = out_x + out_w * (out_y + out_h * (_gx_1 + _gx_0 * c));
        if (indexes(offset) == kx + kw * ky) {{
            gx += gy(offset);
        }}
    }}
}}
""",
        name=f"max_pool_bwd",
    )


@_register_kernel("max_pool_bwd")
def max_pool_bwd(elementwise_kernel, args):
    gy, indexes, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, gx = args
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "kh": kh,
        "kw": kw,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
    }
    max_pool_bwd_kernel()(gy, indexes, gx, uniforms=uniforms)
    return gx


@cache
def avg_pool_fwd_kernel():
    return ElementwiseKernel(
        in_params="raw T inx",
        out_params="T oimg",
        uniforms="i32 h, i32 w, i32 out_h, i32 out_w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw, f32 coeff",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let out_h: i32 = cmeta.out_h;
let out_w: i32 = cmeta.out_w;
let kh: i32 = cmeta.kh;
let kw: i32 = cmeta.kw;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let coeff: f32 = cmeta.coeff;
var c0: i32 = _oimg_0 * cmeta._oimg_shape_1 + _oimg_1;
var out_y: i32 = _oimg_2;
var out_x: i32 = _oimg_3;
var in_y_0: i32 = max(0, out_y * sy - ph);
var in_y_1: i32 = min(h, out_y * sy + kh - ph);
var in_x_0: i32 = max(0, out_x * sx - pw);
var in_x_1: i32 = min(w, out_x * sx + kw - pw);

var val: T = T(0);
for (var yi: i32 = 0; yi < kh; yi++) {{
    var y: i32 = in_y_0 + yi;
    if (y >= in_y_1) {{
        break;
    }}
    var offset_y: i32 = w * (y + h * c0);
    for (var xi: i32 = 0; xi < kw; xi++) {{
        var x: i32 = in_x_0 + xi;
        if (x >= in_x_1) {{
            break;
        }}
        val += inx(x + offset_y);
    }}
}}

oimg = val * coeff;
""",
        name=f"avg_pool_fwd",
    )


@_register_kernel("avg_pool_fwd")
def avg_pool_fwd(elementwise_kernel, args):
    x, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, coeff, out = args
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "kh": kh,
        "kw": kw,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "coeff": coeff,
    }
    avg_pool_fwd_kernel()(x, out, uniforms=uniforms)
    return out


@cache
def avg_pool_bwd_kernel():
    return ElementwiseKernel(
        in_params="raw T gy",
        out_params="T gx",
        uniforms="i32 h, i32 w, i32 out_h, i32 out_w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw, f32 coeff",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let out_h: i32 = cmeta.out_h;
let out_w: i32 = cmeta.out_w;
let kh: i32 = cmeta.kh;
let kw: i32 = cmeta.kw;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let coeff: f32 = cmeta.coeff;
var c0: i32 = _gx_0 * cmeta._gx_shape_1 + _gx_1;
var y: i32 = _gx_2 + ph;
var x: i32 = _gx_3 + pw;
var out_y_0: i32 = max(0, (y - kh + sy) / sy);
var out_y_1: i32 = min(out_h, (y + sy) / sy);
var out_x_0: i32 = max(0, (x - kw + sx) / sx);
var out_x_1: i32 = min(out_w, (x + sx) / sx);
var hc0: i32 = out_h * c0;

var val: T = T(0);
for (var yi: i32 = 0; yi < kh; yi++) {{
    var out_y: i32 = out_y_0 + yi;
    if (out_y >= out_y_1) {{
        break;
    }}
    for (var xi: i32 = 0; xi < kw; xi++) {{
        var out_x: i32 = out_x_0 + xi;
        if (out_x >= out_x_1) {{
            break;
        }}
        val += gy(out_x + out_w * (out_y + hc0));
    }}
}}

gx = val * coeff;
""",
        name=f"avg_pool_bwd",
    )


@_register_kernel("avg_pool_bwd")
def avg_pool_bwd(elementwise_kernel, args):
    gy, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, coeff, gx = args
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "kh": kh,
        "kw": kw,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "coeff": coeff,
    }
    avg_pool_bwd_kernel()(gy, gx, uniforms=uniforms)
    return gx


@cache
def im2col_kernel():
    return ElementwiseKernel(
        in_params="raw T img",
        out_params="T col",
        uniforms="i32 h, i32 w, i32 out_h, i32 out_w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw, i32 dy, i32 dx",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let out_h: i32 = cmeta.out_h;
let out_w: i32 = cmeta.out_w;
let kh: i32 = cmeta.kh;
let kw: i32 = cmeta.kw;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let dy: i32 = cmeta.dy;
let dx: i32 = cmeta.dx;
var c: i32 = cmeta._col_shape_1;

var in_y: i32 = _col_2 * dy + _col_4 * sy - ph;
var in_x: i32 = _col_3 * dx + _col_5 * sx - pw;

if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {{
    col = img(in_x + w * (in_y + h * (_col_1 + c * _col_0)));
}} else {{
    col = T(0);
}}
""",
        name="im2col",
    )


@_register_kernel("im2col")
def im2col(elementwise_kernel, args):
    img, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col = args
    assert col.ndim == 6  # batch, channel, kh, kw, out_h, out_w

    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "kh": kh,
        "kw": kw,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "dy": dy,
        "dx": dx,
    }
    im2col_kernel()(img, col, uniforms=uniforms)
    return col


@cache
def col2im_kernel():
    return ElementwiseKernel(
        in_params="raw T col",
        out_params="T img",
        uniforms="i32 h, i32 w, i32 out_h, i32 out_w, i32 kh, i32 kw, i32 sy, i32 sx, i32 ph, i32 pw, i32 dy, i32 dx",
        operation=f"""
let h: i32 = cmeta.h;
let w: i32 = cmeta.w;
let out_h: i32 = cmeta.out_h;
let out_w: i32 = cmeta.out_w;
let kh: i32 = cmeta.kh;
let kw: i32 = cmeta.kw;
let sy: i32 = cmeta.sy;
let sx: i32 = cmeta.sx;
let ph: i32 = cmeta.ph;
let pw: i32 = cmeta.pw;
let dy: i32 = cmeta.dy;
let dx: i32 = cmeta.dx;
var c: i32 = cmeta._img_shape_1;
var c0: i32 = _img_0 * c + _img_1;

img = T(0);

for (var ky: i32 = 0; ky < kh; ky++) {{
    var out_y: i32 = _img_2 + ph - ky * dy;
    if (0 > out_y || out_y >= out_h * sy) {{continue;}}
    if (out_y % sy != 0) {{continue;}}
    out_y /= sy;
    for (var kx: i32 = 0; kx < kw; kx++) {{
        var out_x: i32 = _img_3 + pw - kx * dx;
        if (0 > out_x || out_x >= out_w * sx) {{continue;}}
        if (out_x % sx != 0) {{continue;}}
        out_x /= sx;
        img += col(out_x + out_w * (out_y + out_h * (kx + kw * (ky + kh * c0))));
    }}
}}
""",
        name="col2im",
    )


@_register_kernel("col2im")
def col2im(elementwise_kernel, args):
    col, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, img = args
    # col is col.reduced_view()
    assert img.ndim == 4

    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "kh": kh,
        "kw": kw,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "dy": dy,
        "dx": dx,
    }
    col2im_kernel()(col, img, uniforms=uniforms)
    return img


@cache
def bn_fwd_kernel():
    return ElementwiseKernel(
        in_params="T x, T mean, T inv_std, T gamma, T beta",
        out_params="T y",
        operation="y = gamma * (x - mean) * inv_std + beta",
        name="bn_fwd",
    )


@_register_kernel("bn_fwd")
def bn_fwd(elementwise_kernel, args):
    y = bn_fwd_kernel()(*args)
    return y


@cache
def bn_bwd_kernel_0():
    return ElementwiseKernel(
        in_params="T gy, T x_hat, T ggamma, T gbeta, T inv_m",
        out_params="T gx",
        operation="gx = gy - (x_hat * ggamma + gbeta) * inv_m",
        name="bn_bwd_0",
    )


@cache
def bn_bwd_kernel_1():
    return ElementwiseKernel(
        in_params="T gx_tmp, T gamma, T inv_std",
        out_params="T gx",
        operation="gx = gx_tmp * (gamma * inv_std)",
        name="bn_bwd_2",
    )


@_register_kernel("bn_bwd")
def bn_bwd(elementwise_kernel, args):
    gy, x_hat, gamma, inv_std, ggamma, gbeta, inv_m = args
    # kernel is divieded into 2 kernels because of the following error:
    # The number of storage buffers (9) in the Compute stage exceeds the maximum per-stage limit (8).
    x_tmp = bn_bwd_kernel_0()(gy, x_hat, ggamma, gbeta, inv_m)
    y = bn_bwd_kernel_1()(x_tmp, gamma, inv_std)
    return y


@cache
def sigmoid_fwd_kernel():
    return ElementwiseKernel(
        in_params="T x",
        out_params="T y",
        operation="y = 1.0 / (1.0 + exp(-x))",
        name="sigmoid_fwd",
    )


@_register_kernel("sigmoid_fwd")
def sigmoid_fwd(elementwise_kernel, args):
    y = sigmoid_fwd_kernel()(*args)
    return y


@cache
def sigmoid_bwd_kernel():
    return ElementwiseKernel(
        in_params="T y, T gy",
        out_params="T gx",
        operation="gx = gy * y * (1.0 - y)",
        name="sigmoid_bwd",
    )


@_register_kernel("sigmoid_bwd")
def sigmoid_bwd(elementwise_kernel, args):
    gx = sigmoid_bwd_kernel()(*args)
    return gx


@cache
def div_bwd_kernel_gx0():
    return ElementwiseKernel(
        in_params="T x1, T gy",
        out_params="T gx0",
        operation="gx0 = gy / x1",
        name="div_bwd_gx0",
    )


@cache
def div_bwd_kernel_gx1():
    return ElementwiseKernel(
        in_params="T gx0, T x0, T x1",
        out_params="T gx1",
        operation="gx1 = -gx0 * x0 / x1",
        name="div_bwd_gx1",
    )


@_register_kernel("div_bwd")
def div_bwd(elementwise_kernel, args):
    x0, x1, gy = args
    gx0 = div_bwd_kernel_gx0()(x1, gy)
    gx1 = div_bwd_kernel_gx1()(gx0, x0, x1)
    return gx0, gx1


def mock_elementwise_kernel(elementwise_kernel, args, size=None, block_size=None):
    if elementwise_kernel.name in _kernels:
        return _kernels[elementwise_kernel.name](elementwise_kernel, args)
    else:
        msg = f"""Requested elementwise kernel is not implemented in WebGPU.
{repr({'in_params': elementwise_kernel.in_params, 'out_params': elementwise_kernel.out_params, 'operation': elementwise_kernel.operation, 'name': elementwise_kernel.name, 'reduce_dims': elementwise_kernel.reduce_dims, 'preamble': elementwise_kernel.preamble, 'no_return': elementwise_kernel.no_return, 'return_tuple': elementwise_kernel.return_tuple, 'loop_prep': elementwise_kernel.loop_prep, 'after_loop': elementwise_kernel.after_loop})}
arguments:
"""
        for arg in args:
            msg += f"class={arg.__class__}, shape={getattr(arg, 'shape') if hasattr(arg, 'shape') else 'scalar'}\n"

        raise NotImplementedError(msg)
