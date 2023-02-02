from js import pythonIO
import numpy as np
import cupy as cp
import json
import time

use_gpu = pythonIO.config.device != "cpu"

def run_once(mat_a, mat_b):
    if use_gpu:
        mat_a = cp.asarray(mat_a)
        mat_b = cp.asarray(mat_b)
    mat_c = mat_a @ mat_b
    if use_gpu:
        mat_c = cp.asnumpy(mat_c)

def bench_one(m, n, k):
    mat_a = np.random.rand(m, k).astype(np.float32)
    mat_b = np.random.rand(k, n).astype(np.float32)
    
    print(f"m={m}, n={n}, k={k}, started")
    # warmup
    for i in range(3):
        run_once(mat_a, mat_b)
    
    # run
    n_runs = 10
    time_start = time.time()
    for i in range(n_runs):
        run_once(mat_a, mat_b)
    time_end = time.time()

    time_avg = (time_end - time_start) / n_runs
    if time_avg > 0:
        gflops = m * n * k * 2 / time_avg / 1000000000
    else:
        gflops = "infinity"
    print(f"m={m}, n={n}, k={k}, average time of {n_runs} runs, {time_avg} sec, {gflops} GFLOPS")

sizes = json.loads('['+pythonIO.config.sizes+']')
for size in sizes:
    bench_one(*size)
