from js import pythonIO
import numpy as np
import cupy as cp
import cupyx.scipy
import time

use_gpu = pythonIO.config.use_gpu


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


def run_once(price, strike, t, use_gpu):
    if use_gpu:
        price = cp.asarray(price)
        strike = cp.asarray(strike)
        t = cp.asarray(t)
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
    print(f"samples={samples}, started")
    price, strike, t = generate_input(samples)
    # warmup
    for _ in range(3):
        run_once(price, strike, t, use_gpu)

    # run
    n_runs = 10
    time_start = time.time()
    for _ in range(n_runs):
        run_once(price, strike, t, use_gpu)
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
    ret_cpu = run_once(price, strike, t, False)
    ret_gpu = run_once(price, strike, t, True)
    print("allclose:", np.allclose(ret_cpu, ret_gpu, rtol=1e-2, atol=1e-2))


if pythonIO.config.mode == "benchmark":
    bench_one(pythonIO.config.samples)
elif pythonIO.config.mode == "verify":
    verify(pythonIO.config.samples)
print("done")
