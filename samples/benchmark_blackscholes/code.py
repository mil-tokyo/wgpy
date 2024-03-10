from js import pythonIO
import numpy as np
import cupy as cp
import cupyx.scipy
import time

use_gpu = pythonIO.config.use_gpu
kernel_type = pythonIO.config.kernel_type


def BlackScholes(S, X, T, R, V):
    xp = cp.get_array_module(S)
    xpx = cupyx.scipy.get_array_module(S)
    VqT = V * xp.sqrt(T)
    d1 = (xp.log(S / X) + (R + 0.5 * V * V) * T) / VqT
    d2 = d1 / VqT
    del VqT
    n1 = 0.5 + 0.5 * xpx.special.erf(d1 * (1.0 / np.sqrt(2.0)))
    del d1
    n2 = 0.5 + 0.5 * xpx.special.erf(d2 * (1.0 / np.sqrt(2.0)))
    del d2
    eRT = xp.exp(-R * T)
    return S * n1 - X * eRT * n2


### begin custom kernel

backend = cp.get_backend_name()
if backend == "webgpu":
    from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
elif backend == "webgl":
    from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

_bs_kernel = None


def customBlackScholes(S, X, T, R, V):
    global _bs_kernel
    if backend == "webgpu":
        if _bs_kernel is None:
            _bs_kernel = ElementwiseKernel(
                in_params="f32 s, f32 x, f32 t, f32 r, f32 v",
                out_params="f32 y",
                preamble="""
fn erf(x: f32) -> f32 {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    let absx = abs(x);
    let t = 1.0 / (1.0 + p * absx);
    let z = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-absx * absx);

    var y: f32;
    if (x < 0.0) {
        y = -z;
    } else {
        y = z;
    }
    return y;
}
""",
                operation="""
let VqT = v * sqrt(t);
let d1 = (log(s / x) + (r + 0.5 * v * v) * t) / VqT;
let d2 = d1 / VqT;
let n1 = 0.5 + 0.5 * erf(d1 * (1.0 / sqrt(2.0)));
let n2 = 0.5 + 0.5 * erf(d2 * (1.0 / sqrt(2.0)));
let eRT = exp(-r * t);
y = s * n1 - x * eRT * n2;
                """,
                name="black_scholes",
            )
        return _bs_kernel(S, X, T, R, V)
    elif backend == "webgl":
        if _bs_kernel is None:
            _bs_kernel = ElementwiseKernel(
                in_params="float s, float x, float t, float r, float v",
                out_params="float y",
                preamble="""
float erf(float x) {
    float a1 = 0.254829592;
    float a2 = -0.284496736;
    float a3 = 1.421413741;
    float a4 = -1.453152027;
    float a5 = 1.061405429;
    float p = 0.3275911;

    float sign = x < 0.0 ? -1.0 : 1.0;
    float absx = abs(x);
    float t = 1.0 / (1.0 + p * absx);
    float z = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-absx * absx);

    float y = sign * z;
    return y;
}
""",
                operation="""
float VqT = v * sqrt(t);
float d1 = (log(s / x) + (r + 0.5 * v * v) * t) / VqT;
float d2 = d1 / VqT;
float n1 = 0.5 + 0.5 * erf(d1 * (1.0 / sqrt(2.0)));
float n2 = 0.5 + 0.5 * erf(d2 * (1.0 / sqrt(2.0)));
float eRT = exp(-r * t);
y = s * n1 - x * eRT * n2;
                """,
                name="black_scholes",
            )
        return _bs_kernel(S, X, T, R, V)
    raise ValueError


_nop_kernel = None


def customNOP(S, X, T, R, V):
    global _nop_kernel
    if backend == "webgpu":
        if _nop_kernel is None:
            _nop_kernel = ElementwiseKernel(
                in_params="f32 s, f32 x, f32 t, f32 r, f32 v",
                out_params="f32 y",
                operation="""
y = 1.0;
                """,
                name="nop",
            )
        return _nop_kernel(S, X, T, R, V)
    elif backend == "webgl":
        if _nop_kernel is None:
            _nop_kernel = ElementwiseKernel(
                in_params="float s, float x, float t, float r, float v",
                out_params="float y",
                operation="""
y = 1.0;
                """,
                name="nop",
            )
        return _nop_kernel(S, X, T, R, V)
    raise ValueError


### end custom kernel


def run_once(price, strike, t, use_gpu, kernel_type):
    if use_gpu:
        price = cp.asarray(price)
        strike = cp.asarray(strike)
        t = cp.asarray(t)
    if use_gpu:
        if kernel_type == "normal":
            ret = BlackScholes(price, strike, t, 0.1, 0.2)
        elif kernel_type == "custom":
            ret = customBlackScholes(price, strike, t, 0.1, 0.2)
        elif kernel_type == "nop":
            ret = customNOP(price, strike, t, 0.1, 0.2)
        else:
            raise ValueError
    else:
        ret = BlackScholes(price, strike, t, 0.1, 0.2)
    if use_gpu:
        ret = cp.asnumpy(ret)
    return ret


def generate_input(samples: int):
    price = np.random.uniform(10.0, 50.0, samples).astype(np.float32)
    strike = np.random.uniform(10.0, 50.0, samples).astype(np.float32)
    t = np.random.uniform(1.0, 2.0, samples).astype(np.float32)
    return price, strike, t


def bench_one(samples: int):
    print(f"samples={samples}, kernel={kernel_type}, started")
    price, strike, t = generate_input(samples)
    # warmup
    for _ in range(3):
        run_once(price, strike, t, use_gpu, kernel_type)

    # run
    n_runs = 10
    time_start = time.time()
    for _ in range(n_runs):
        run_once(price, strike, t, use_gpu, kernel_type)
    time_end = time.time()

    time_avg = (time_end - time_start) / n_runs
    if time_avg > 0:
        samples_per_sec = samples / time_avg
    else:
        samples_per_sec = "infinity"
    print(
        f"samples={samples}, average time of {n_runs} runs, {time_avg} sec, {samples_per_sec} samples/sec"
    )


def verify(samples: int):
    print(f"verify samples={samples}, started")
    price, strike, t = generate_input(samples)
    ret_cpu = run_once(price, strike, t, False, kernel_type)
    ret_gpu = run_once(price, strike, t, True, kernel_type)
    print("allclose:", np.allclose(ret_cpu, ret_gpu, rtol=1e-2, atol=1e-2))


if pythonIO.config.mode == "benchmark":
    bench_one(pythonIO.config.samples)
elif pythonIO.config.mode == "verify":
    verify(pythonIO.config.samples)
print("done")
