from functools import cache
import numpy as np
from wgpy.construct import asarray, asnumpy
from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

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
        in_params="float y, float gy",
        out_params="float gx",
        operation="gx = y > 0.0 ? gy : 0.0",
        name="relu_bwd",
    )


@_register_kernel("relu_bwd")
def relu_bwd(elementwise_kernel, args):
    y, gy = args
    gx = relu_bwd_kernel()(y, gy)
    return gx


@cache
def max_pool_fwd_kernel_indexes_for_shape(loop_max):
    return ElementwiseKernel(
        in_params="raw T inx",
        out_params="int indexes",
        uniforms="int h, int w, int sy, int sx, int ph, int pw",
        preamble=f"""
#define kh {loop_max[0]}
#define kw {loop_max[1]}
""",
        operation=f"""
int c = _indexes_shape_1;
int out_y = _indexes_2;
int out_x = _indexes_3;
int in_y_0 = max(0, out_y * sy - ph);
int in_y_1 = min(h, out_y * sy + kh - ph);
int in_x_0 = max(0, out_x * sx - pw);
int in_x_1 = min(w, out_x * sx + kw - pw);

T maxval = inx(((_indexes_0 * c + _indexes_1) * h + in_y_0) * w + in_x_0);
int argmax_y = in_y_0;
int argmax_x = in_x_0;
for (int yi = 0; yi < kh; yi++) {{
    int y = in_y_0 + yi;
    if (y >= in_y_1) {{
        break;
    }}
    for (int xi = 0; xi < kw; xi++) {{
        int x = in_x_0 + xi;
        if (x >= in_x_1) {{
            break;
        }}
        T v = inx(((_indexes_0 * c + _indexes_1) * h + y) * w + x);
        if (maxval < v) {{
            maxval = v;
            argmax_y = y;
            argmax_x = x;
        }}
    }}
}}

int argmax_ky = argmax_y + ph - out_y * sy;
int argmax_kx = argmax_x + pw - out_x * sx;
indexes = argmax_kx + kw * argmax_ky;
""",
        name=f"max_pool_fwd_indexes_{loop_max[0]}_{loop_max[1]}",
    )


@cache
def max_pool_fwd_kernel_maxval():
    return ElementwiseKernel(
        in_params="raw T inx, S indexes",
        out_params="T maxval",
        uniforms="int h, int w, int sy, int sx, int ph, int pw, int kw",
        operation=f"""
int c = _maxval_shape_1;
int out_y = _maxval_2;
int out_x = _maxval_3;

int tmp = indexes;
int argmax_ky = tmp / kw;
int argmax_kx = tmp - kw * argmax_ky;
int argmax_y = argmax_ky + out_y * sy - ph;
int argmax_x = argmax_kx + out_x * sx - pw;

maxval = inx(((_maxval_0 * c + _maxval_1) * h + argmax_y) * w + argmax_x);
""",
        name="max_pool_fwd_maxval",
    )


@_register_kernel("max_pool_fwd")
def max_pool_fwd(elementwise_kernel, args):
    x, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, y, indexes = args
    loop_max = (kh, kw)
    uniforms = {
        "h": h,
        "w": w,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
    }
    max_pool_fwd_kernel_indexes_for_shape(loop_max)(x, indexes, uniforms=uniforms)
    uniforms["kw"] = kw
    max_pool_fwd_kernel_maxval()(x, indexes, y, uniforms=uniforms)
    return y, indexes


@cache
def max_pool_bwd_kernel_for_shape(loop_max):
    return ElementwiseKernel(
        in_params="raw T gy, raw int indexes",
        out_params="T gx",
        uniforms="int h, int w, int out_h, int out_w, int sy, int sx, int ph, int pw",
        preamble=f"""
#define kh {loop_max[0]}
#define kw {loop_max[1]}
""",
        operation=f"""
int c = _gx_shape_1;
int y = _gx_2 + ph;
int x = _gx_3 + pw;
int out_y_0 = max(0, (y - kh + sy) / sy);
int out_y_1 = min(out_h, (y + sy) / sy);
int out_x_0 = max(0, (x - kw + sx) / sx);
int out_x_1 = min(out_w, (x + sx) / sx);

gx = T(0);
for (int yi = 0; yi < kh; yi++) {{
    int out_y = out_y_0 + yi;
    if (out_y >= out_y_1) {{
        break;
    }}
    int ky = y - out_y * sy;
    for (int xi = 0; xi < kw; xi++) {{
        int out_x = out_x_0 + xi;
        if (out_x >= out_x_1) {{
            break;
        }}
        int kx = x - out_x * sx;
        int offset = out_x + out_w * (out_y + out_h * (_gx_1 + _gx_0 * c));
        if (indexes(offset) == kx + kw * ky) {{
            gx += gy(offset);
        }}
    }}
}}
""",
        name=f"max_pool_bwd_{loop_max[0]}_{loop_max[1]}",
    )


@_register_kernel("max_pool_bwd")
def max_pool_bwd(elementwise_kernel, args):
    gy, indexes, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, gx = args
    loop_max = (kh, kw)
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
    }
    max_pool_bwd_kernel_for_shape(loop_max)(gy, indexes, gx, uniforms=uniforms)
    return gx


@cache
def avg_pool_fwd_kernel_for_shape(loop_max):
    return ElementwiseKernel(
        in_params="raw T inx",
        out_params="T oimg",
        uniforms="int h, int w, int out_h, int out_w, int sy, int sx, int ph, int pw, float coeff",
        preamble=f"""
#define kh {loop_max[0]}
#define kw {loop_max[1]}
""",
        operation=f"""
int c0 = _oimg_0 * _oimg_shape_1 + _oimg_1;
int out_y = _oimg_2;
int out_x = _oimg_3;
int in_y_0 = max(0, out_y * sy - ph);
int in_y_1 = min(h, out_y * sy + kh - ph);
int in_x_0 = max(0, out_x * sx - pw);
int in_x_1 = min(w, out_x * sx + kw - pw);

T val = T(0);
for (int yi = 0; yi < kh; yi++) {{
    int y = in_y_0 + yi;
    if (y >= in_y_1) {{
        break;
    }}
    int offset_y = w * (y + h * c0);
    for (int xi = 0; xi < kw; xi++) {{
        int x = in_x_0 + xi;
        if (x >= in_x_1) {{
            break;
        }}
        val += inx(x + offset_y);
    }}
}}

oimg = val * coeff;
""",
        name=f"avg_pool_fwd_{loop_max[0]}_{loop_max[1]}",
    )


@_register_kernel("avg_pool_fwd")
def avg_pool_fwd(elementwise_kernel, args):
    x, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, coeff, out = args
    loop_max = (kh, kw)
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "coeff": coeff,
    }
    avg_pool_fwd_kernel_for_shape(loop_max)(x, out, uniforms=uniforms)
    return out


@cache
def avg_pool_bwd_kernel_for_shape(loop_max):
    return ElementwiseKernel(
        in_params="raw T gy",
        out_params="T gx",
        uniforms="int h, int w, int out_h, int out_w, int sy, int sx, int ph, int pw, float coeff",
        preamble=f"""
#define kh {loop_max[0]}
#define kw {loop_max[1]}
""",
        operation=f"""
int c0 = _gx_0 * _gx_shape_1 + _gx_1;
int y = _gx_2 + ph;
int x = _gx_3 + pw;
int out_y_0 = max(0, (y - kh + sy) / sy);
int out_y_1 = min(out_h, (y + sy) / sy);
int out_x_0 = max(0, (x - kw + sx) / sx);
int out_x_1 = min(out_w, (x + sx) / sx);
int hc0 = out_h * c0;

T val = T(0);
for (int yi = 0; yi < kh; yi++) {{
    int out_y = out_y_0 + yi;
    if (out_y >= out_y_1) {{
        break;
    }}
    for (int xi = 0; xi < kw; xi++) {{
        int out_x = out_x_0 + xi;
        if (out_x >= out_x_1) {{
            break;
        }}
        val += gy(out_x + out_w * (out_y + hc0));
    }}
}}

gx = val * coeff;
""",
        name=f"avg_pool_bwd_{loop_max[0]}_{loop_max[1]}",
    )


@_register_kernel("avg_pool_bwd")
def avg_pool_bwd(elementwise_kernel, args):
    gy, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, coeff, gx = args
    loop_max = (kh, kw)
    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "coeff": coeff,
    }
    avg_pool_bwd_kernel_for_shape(loop_max)(gy, gx, uniforms=uniforms)
    return gx


@cache
def im2col_kernel():
    return ElementwiseKernel(
        in_params="raw T img",
        out_params="T col",
        uniforms="int h, int w, int out_h, int out_w, int kh, int kw, int sy, int sx, int ph, int pw, int dy, int dx",
        operation=f"""
int c = _col_shape_1;

int in_y = _col_2 * dy + _col_4 * sy - ph;
int in_x = _col_3 * dx + _col_5 * sx - pw;

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
def col2im_kernel(kernel_key):
    kh, kw = kernel_key
    return ElementwiseKernel(
        in_params="raw T col",
        out_params="T img",
        uniforms="int h, int w, int out_h, int out_w, int sy, int sx, int ph, int pw, int dy, int dx",
        preamble=f"""
#define kh {kh}
#define kw {kw}
""",
        operation=f"""
int c = _img_shape_1;
int c0 = _img_0 * c + _img_1;

img = T(0);

for (int ky = 0; ky < kh; ky++) {{
    int out_y = _img_2 + ph - ky * dy;
    if (0 > out_y || out_y >= out_h * sy) continue;
    if (out_y % sy != 0) continue;
    out_y /= sy;
    for (int kx = 0; kx < kw; kx++) {{
        int out_x = _img_3 + pw - kx * dx;
        if (0 > out_x || out_x >= out_w * sx) continue;
        if (out_x % sx != 0) continue;
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
    kernel_key = (kh, kw)

    uniforms = {
        "h": h,
        "w": w,
        "out_h": out_h,
        "out_w": out_w,
        "sy": sy,
        "sx": sx,
        "ph": ph,
        "pw": pw,
        "dy": dy,
        "dx": dx,
    }
    col2im_kernel(kernel_key)(col, img, uniforms=uniforms)
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
def bn_bwd_kernel():
    return ElementwiseKernel(
        in_params="T gy, T x_hat, T gamma, T inv_std, T ggamma, T gbeta, T inv_m",
        out_params="T gx",
        operation="gx = (gamma * inv_std) * (gy - (x_hat * ggamma + gbeta) * inv_m)",
        name="bn_bwd",
    )


@_register_kernel("bn_bwd")
def bn_bwd(elementwise_kernel, args):
    y = bn_bwd_kernel()(*args)
    return y


@cache
def sigmoid_fwd_kernel():
    # Chainer's implementation uses tanh(x * 0.5) * 0.5 + 0.5, but tanh produces NaN for large x in some GPU, so we avoid using it.
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
    y = sigmoid_bwd_kernel()(*args)
    return y


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
        msg = f"""Requested elementwise kernel is not implemented in WebGL.
{repr({'in_params': elementwise_kernel.in_params, 'out_params': elementwise_kernel.out_params, 'operation': elementwise_kernel.operation, 'name': elementwise_kernel.name, 'reduce_dims': elementwise_kernel.reduce_dims, 'preamble': elementwise_kernel.preamble, 'no_return': elementwise_kernel.no_return, 'return_tuple': elementwise_kernel.return_tuple, 'loop_prep': elementwise_kernel.loop_prep, 'after_loop': elementwise_kernel.after_loop})}
arguments:
"""
        for arg in args:
            msg += f"class={arg.__class__}, shape={getattr(arg, 'shape') if hasattr(arg, 'shape') else 'scalar'}\n"

        raise NotImplementedError(msg)
