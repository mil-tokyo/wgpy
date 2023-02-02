importScripts('../../lib/pyodide/pyodide.js');
importScripts('../../dist/wgpy-worker.js');

let pyodide;

function log(message) {
  postMessage({ namespace: 'app', method: 'log', message: message });
}

async function loadPythonCode() {
  const f = await fetch('code.py');
  if (!f.ok) {
    throw new Error(f.statusText);
  }
  return f.text();
}

async function start(data) {
  log('Initializing wgpy worker-side javascript interface');
  await wgpy.initWorker();

  log('Loading pyodide');
  pyodide = await loadPyodide({
    indexURL: '../../lib/pyodide/',
  });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  await pyodide.loadPackage('../../dist/wgpy_webgl-1.0.0-py3-none-any.whl');

  log('Loading pyodide succeeded');
  const pythonCode = await loadPythonCode();

  self.pythonIO = {
    getInputData: () => JSON.stringify(data),
    setOutputData: (data) =>
      postMessage({
        namespace: 'app',
        method: 'output',
        data: JSON.parse(data),
      }),
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
      start(ev.data.data).catch((reason) => log(`Worker error: ${reason}`));
      break;
  }
});
