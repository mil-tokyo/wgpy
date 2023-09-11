importScripts('/lib/pyodide/pyodide.js');
importScripts('/dist/wgpy-worker.js');

let pyodide;

function log(message) {
  postMessage({ namespace: 'app', method: 'log', message: message });
}

async function loadPythonCode() {
  const f = await fetch('run_test.py');
  if (!f.ok) {
    throw new Error(f.statusText);
  }
  return f.text();
}

async function start(backend, testPath) {
  log('Initializing wgpy worker-side javascript interface');
  await wgpy.initWorker();

  log(`Loading pyodide with wgpy (backend: ${backend})`);
  pyodide = await loadPyodide({
    indexURL: '/lib/pyodide/',
    stdout: log,
    stderr: log,
  });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  await pyodide.loadPackage('scipy');
  await pyodide.loadPackage('pytest');
  await pyodide.loadPackage(`/dist/wgpy_${backend}-1.0.0-py3-none-any.whl`);
  await pyodide.loadPackage('/dist/wgpy_test-1.0.0-py3-none-any.whl');
  // loadPackage of custom wheel does not install dependencies. In contrast, micropip.install does so.
  // await pyodide.loadPackage('/lib/chainer-5.4.0-py3-none-any.whl');

  log('Loading pyodide succeeded');
  const pythonCode = await loadPythonCode();

  self.pythonIO = {
    testPath,
  };
  log('Running python code');
  await pyodide.runPythonAsync(pythonCode);
}

addEventListener('message', (ev) => {
  if (ev.data.namespace !== 'app') {
    // message for library
    return;
  }

  switch (ev.data.method) {
    case 'start':
      start(ev.data.backend, ev.data.testPath).catch((reason) => log(`Worker error: ${reason}`));
      break;
  }
});
