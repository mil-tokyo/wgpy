# wgpy: WebGL accelerated numpy-compatible array library for web browser

wgpy is a WebGL accelerated numpy-compatible array library for web browsers. It runs on [Pyodide](https://pyodide.org/), the python runtime which runs on the web browser. Deep learning can also be performed on GPUs in conjunction with [Chainer](https://github.com/chainer/chainer).

# Setup Environment

Python 3.10 and Node.js 16 are required.

## Downloading Pyodide

```
./scripts/download-pyodide.sh
```

## Install dependencies

```
pip install -r requirements.txt
npm install
```

# Build

```
npm run build
```

The output is placed in the `dist` directory.
The Pyodide is expected to work on WebWorker, not the main thread, because wgpy is mainly forcing for running heavy tasks. 
`{wgpy-main.js,wgpy-worker.js}` are the JavaScript library. The `wgpy-main.js` needs to be loaded in the main thread (using `<script>` tag). It calls the WebGL API following the commands from WebWorker. `wgpy-worker.js` has to be loaded in WebWorker thread (using `importScripts`). This script exposes an interface to the wgpy python library and proxy it to the main thread. `wgpy_webgl-<version>-py3-none-any.whl` is the python library that is loaded using `await pyodide.loadPackage('path/to/wgpy_webgl-<version>-py3-none-any.whl');` in the JavaScript code or `micropip.install('path/to/wgpy_webgl-<version>-py3-none-any.whl')` in the Python code. It gives a python API that has the same interface as numpy. The wgpy can be imported as `import wgpy as wp`. A wrapper of wgpy to provide the same API as cupy is also implemented to minimize changes to Chainer (e.g. `import cupy as cp`). `wgpy_test-<version>-py3-none-any.whl` contains test code.

Here is the minimum sample code:

```python
import wgpy as wp
# transfer array to GPU
gpu_in_data = wp.asarray([1.0, 2.0, 2.5])
# compute on GPU
gpu_out_data = gpu_in_data * 2.0
# transfer array to CPU (numpy ndarray)
out_data = wp.asnumpy(gpu_out_data)
```

# Run Examples

Start HTTP server:

```
npm start
```

Open http://localhost:8000 and click link to a sample.

If you access to the server from another device such as smartphones, you need to access the server by a URL starting with `https://`. Even if a URL such as `http://192.168.0.2` seems to display the page, the logic does not work. To access to the server via https URL, you can use online service such as [ngrok](https://ngrok.com/).

After you install ngrok, run the command in a new terminal:

```
ngrok http 8000
```

Then access to the displayed URL (`https://*.ngrok.io`).

The reason why URLs beginning with https are required is that secure origin is needed to use the SharedArrayBuffer that this library uses internally.

# Test

## Build

```
npm run build:test
```

# Run

```
npm start
```

Open <http://localhost:8000/test/> and click TEST button. If you run only one test file, specify it in the URL such as <http://localhost:8000/test/?test_path=/test_dot.py> .

# About samples which uses Chainer

To run MNIST sample, you need the following steps.

## Creating a wheel with Chainer modified for Pyodide

```
./scripts/make-chainer-wheel.sh
```

`lib/chainer-5.4.0-py3-none-any.whl` is generated.

Chainer 5.4.0 is the last version of Chainer which contains pure-python code only. Extra work is needed to support Chainer versions with native (C++) extensions in Pyodide.

## Download and preprocess MNIST dataset

Chainer 5.4.0 is needed.

```
pip install chainer==5.4.0
```

Run:

```
python scripts/pack_mnist.py
```

`lib/mnist.zip` is generated.
