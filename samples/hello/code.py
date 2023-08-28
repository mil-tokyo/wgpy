import json
import numpy as np
import wgpy as cp
from js import pythonIO

print("backend name:", cp.get_backend_name())

in_data = np.array(json.loads(pythonIO.getInputData()), dtype=np.float32)

print("input: ", in_data)

gpu_in_data = cp.asarray(in_data)

gpu_out_data = gpu_in_data * 2

out_data = cp.asnumpy(gpu_out_data)

print("output: ", out_data)

pythonIO.setOutputData(json.dumps(out_data.tolist()))
