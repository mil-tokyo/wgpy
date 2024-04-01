from js import pythonIO
import numpy as np
import json
import time

use_gpu = False
try:
    import cupy as cp

    use_gpu = True
except:
    print("cupy is not available")


def run_once(mat_a, mat_b, times, use_gpu):
    if use_gpu:
        mat_a = cp.asarray(mat_a)
        mat_b = cp.asarray(mat_b)
    mat_c = None
    for _ in range(times):
        if mat_c is None:
            mat_c = mat_a @ mat_b
        else:
            mat_c += mat_a @ mat_b
    if use_gpu:
        mat_c = cp.asnumpy(mat_c)
    return mat_c


def bench_one(m, n, k):
    mat_a = np.random.rand(m, k).astype(np.float32)
    mat_b = np.random.rand(k, n).astype(np.float32)
    times = pythonIO.config.times

    print(f"m={m}, n={n}, k={k}, started")
    # warmup
    for i in range(3):
        ret = run_once(mat_a, mat_b, times, use_gpu)
        if i == 0 and pythonIO.config.compareResult:
            ret_cpu = run_once(mat_a, mat_b, times, use_gpu=False)
            max_diff = np.max(np.abs(ret - ret_cpu))
            print(f"max diff between cpu and gpu: {max_diff}")
            del ret_cpu
        del ret

    # run
    n_runs = 10
    elapseds = []
    time_start = time.time()
    for i in range(n_runs):
        run_once(mat_a, mat_b, times, use_gpu)
        time_end = time.time()
        elapseds.append(time_end - time_start)
        time_start = time_end

    time_avg = np.mean(elapseds)
    time_std = np.std(elapseds)
    if time_avg > 0:
        gflops = m * n * k * 2 * times / time_avg / 1000000000
    else:
        gflops = "infinity"
    print(
        f"m={m}, n={n}, k={k}, {times} multiplications. Statistics of {n_runs} runs: {time_avg:.3g}+-{time_std:.3g} sec / run, {gflops} GFLOPS"
    )


sizes = json.loads("[" + pythonIO.config.sizes + "]")
for size in sizes:
    bench_one(*size)
print("done")
