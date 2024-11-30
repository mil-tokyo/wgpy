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
for(var k: u32 = 0u; k < """+str(n_iters)+"""u; k = k + 1u) {
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
for (int k = 0; k < """+str(n_iters)+"""; k++) {
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


_nop_kernel = None


def customNOP(R, I, n_iters):
    global _nop_kernel
    if backend == "webgpu":
        if _nop_kernel is None:
            _nop_kernel = ElementwiseKernel(
                in_params="f32 real, f32 imag",
                out_params="i32 c",
                operation="""
c = 1;
                """,
                name="nop",
            )
        return _nop_kernel(R, I)
    elif backend == "webgl":
        if _nop_kernel is None:
            _nop_kernel = ElementwiseKernel(
                in_params="float real, float imag",
                out_params="int c",
                operation="""
c = 1;
                """,
                name="nop",
            )
        return _nop_kernel(R, I)
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
        elif kernel_type == "nop":
            ret = customNOP(real, imag,n_iters)
        else:
            raise ValueError
    else:
        ret = mandelbrot(real, imag, n_iters)
    if use_gpu:
        ret = cp.asnumpy(ret)
    return ret


def generate_input(grid: int):
    real = np.linspace(-2.0, 0.5, grid, dtype=np.float32)[np.newaxis, :]
    imag = np.linspace(-1.2, 1.2, grid, dtype=np.float32)[:, np.newaxis]
    return real, imag


def bench_one(grid: int):
    print(f"grid={grid}, kernel={kernel_type}, started")
    real, imag = generate_input(grid)
    # warmup
    for _ in range(3):
        run_once(real, imag, use_gpu, kernel_type)

    # run
    n_runs = 10
    times = []
    for _ in range(n_runs):
        time_start = time.time()
        run_once(real, imag, use_gpu, kernel_type)
        time_end = time.time()
        times.append(time_end - time_start)

    time_avg = np.mean(times)
    if time_avg > 0:
        samples_per_sec = (grid * grid) / time_avg
    else:
        samples_per_sec = "infinity"
    print(
        f"grid={grid}, average time of {n_runs} runs, {time_avg} sec, {samples_per_sec} items/sec"
    )
    print("std", np.std(times), "min", np.min(times), "max", np.max(times))
    print(get_performance_metrics())


def verify(grid: int):
    print(f"verify grid={grid}, started")
    real, imag = generate_input(grid)
    ret_cpu = run_once(real, imag, False, kernel_type)
    ret_gpu = run_once(real, imag, True, kernel_type)
    count_nonzero = np.count_nonzero(ret_cpu != ret_gpu)
    print(f"number of different output: {count_nonzero} ({int(count_nonzero / ret_cpu.size * 100)} % of {ret_cpu.size} items)", )


def visualize(grid: int):
    print(f"visualize grid={grid}, started")
    real, imag = generate_input(grid)
    ret_gpu = run_once(real, imag, True, kernel_type)
    display_image(count_to_img(ret_gpu))


def count_to_img(count):
    # TODO: improve visualization with colormap
    f = count.astype(np.float32)
    f = f / f.max()
    return Image.fromarray(np.repeat((f * 255.0)[...,np.newaxis].astype(np.uint8), 3, axis=2))


def display_image(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    png_bin = img_bytes.getvalue()
    # to base64 data url
    import base64
    png_b64 = base64.b64encode(png_bin).decode('ascii')
    pythonIO.displayImage(f'data:image/png;base64,{png_b64}')


if pythonIO.config.mode == "benchmark":
    bench_one(pythonIO.config.grid)
elif pythonIO.config.mode == "verify":
    verify(pythonIO.config.grid)
elif pythonIO.config.mode == "visualize":
    visualize(pythonIO.config.grid)
print("done")
