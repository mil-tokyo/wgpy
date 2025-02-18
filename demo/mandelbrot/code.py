from js import pythonIO
import numpy as np
import cupy as cp
import cupyx.scipy
import time

# for visualization
from PIL import Image
import io
import base64

use_gpu = pythonIO.config.use_gpu
kernel_type = pythonIO.config.kernel_type
np.random.seed(0)

n_iters = 500


def mandelbrot(real, imag, n_iters):
    xp = cp.get_array_module(real)
    xs = xp.zeros((real.size, imag.size), dtype=np.float32)
    ys = xp.zeros((real.size, imag.size), dtype=np.float32)
    count = xp.zeros((real.size, imag.size), dtype=np.int32)
    for _ in range(n_iters):
        xs, ys = xs * xs - ys * ys + real, xs * ys * 2.0 + imag
        count += ((xs * xs + ys * ys) < 4.0).astype(np.int32)
    return count


### begin custom kernel

backend = cp.get_backend_name()
if backend == "webgpu":
    from wgpy_backends.webgpu import get_performance_metrics
    from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
elif backend == "webgl":
    from wgpy_backends.webgl import get_performance_metrics
    from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

_bs_kernels = {}


def customMandelbrot(R, I, n_iters):
    if backend == "webgpu":
        if (_bs_kernel := _bs_kernels.get(n_iters)) is None:
            _bs_kernel = ElementwiseKernel(
                in_params="f32 real, f32 imag",
                out_params="i32 c",
                operation="""
c = 0;
var x: f32 = 0.0;
var y: f32 = 0.0;
for(var k: u32 = 0u; k < """
                + str(n_iters)
                + """u; k = k + 1u) {
    var nx: f32 = x * x - y * y + real;
    var ny: f32 = x * y * 2.0 + imag;
    x = nx;
    y = ny;
    if (x * x + y * y < 4.0) {
        c = c + 1;
    }
}
                """,
                name=f"mandelbrot_{n_iters}",
            )
            _bs_kernels[n_iters] = _bs_kernel
        return _bs_kernel(R, I)
    elif backend == "webgl":
        if (_bs_kernel := _bs_kernels.get(n_iters)) is None:
            _bs_kernel = ElementwiseKernel(
                in_params="float real, float imag",
                out_params="int c",
                operation="""
c = 0;
float x = 0.0;
float y = 0.0;
for (int k = 0; k < """
                + str(n_iters)
                + """; k++) {
    float nx = x * x - y * y + real;
    float ny = x * y * 2.0 + imag;
    x = nx;
    y = ny;
    if (x * x + y * y < 4.0) {
        c = c + 1;
    }
}
                """,
                name=f"mandelbrot_{n_iters}",
            )
            _bs_kernels[n_iters] = _bs_kernel
        return _bs_kernel(R, I)
    raise ValueError


### end custom kernel


def run_once(real, imag, use_gpu, kernel_type):
    if use_gpu:
        real = cp.asarray(real)
        imag = cp.asarray(imag)
    if use_gpu:
        if kernel_type == "normal":
            ret = mandelbrot(real, imag, n_iters)
        elif kernel_type == "custom":
            ret = customMandelbrot(real, imag, n_iters)
        else:
            raise ValueError
    else:
        ret = mandelbrot(real, imag, n_iters)
    if use_gpu:
        ret = cp.asnumpy(ret)
    return ret


def generate_input(grid: int, real_min=-2.0, real_max=0.5, imag_min=-1.2, imag_max=1.2):
    real = np.linspace(real_min, real_max, grid, dtype=np.float32)[np.newaxis, :]
    imag = np.linspace(imag_min, imag_max, grid, dtype=np.float32)[:, np.newaxis]
    return real, imag


def visualize(grid: int, real_min=-2.0, real_max=0.5, imag_min=-1.2, imag_max=1.2):
    print(f"visualizeing grid={grid}")
    real, imag = generate_input(grid, real_min, real_max, imag_min, imag_max)
    start_time = time.time()
    ret_gpu = run_once(real, imag, use_gpu, kernel_type)
    end_time = time.time()
    print(f"elapsed time: {end_time - start_time:.3f} sec")
    display_image(count_to_img(ret_gpu))


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=1)


def generate_color_map():
    hsv = np.array(
        [
            np.linspace(0.66, 0.0, n_iters + 1),
            np.full(n_iters + 1, 0.9),
            np.full(n_iters + 1, 0.95),
        ]
    ).T
    rgb = hsv_to_rgb(hsv)
    rgb[n_iters] = [1, 1, 1]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


color_map = generate_color_map()


def count_to_img(count):
    # TODO: improve visualization with colormap
    # f = count.astype(np.float32)
    # f = f / f.max()
    # return Image.fromarray(
    #     np.repeat((f * 255.0)[..., np.newaxis].astype(np.uint8), 3, axis=2)
    # )
    return Image.fromarray(color_map[count])


def display_image(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG", compress_level=0)
    png_bin = img_bytes.getvalue()
    # to base64 data url
    import base64

    png_b64 = base64.b64encode(png_bin).decode("ascii")
    pythonIO.displayImage(f"data:image/png;base64,{png_b64}")
