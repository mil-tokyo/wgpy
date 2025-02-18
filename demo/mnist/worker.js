importScripts('../../lib/pyodide/pyodide.js');
importScripts('../../dist/wgpy-worker.js');

let pyodide;

function log(message) {
  postMessage({ namespace: 'app', method: 'log', message: message });
}

function stdout(line) {
  // remove escape seqeunce
  log(line.replace(/\x1b\[[0-9;]*[A-Za-z]/, ''));
}

async function loadPythonCode() {
  const f = await fetch('code.py');
  if (!f.ok) {
    throw new Error(f.statusText);
  }
  return f.text();
}

async function start(config) {
  log('Initializing wgpy worker-side javascript interface');
  let initWorkerResult = null;
  try {
    initWorkerResult = await wgpy.initWorker();    
  } catch (e) {
    // if no backend is available, wgpy.initWorker throws an error.
    log(`initWorker failed: ${e.message}`);
  }

  log('Loading pyodide');
  pyodide = await loadPyodide({
    indexURL: '../../lib/pyodide/',
    stdout: stdout,
    stderr: stdout,
  });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  if (initWorkerResult) {
    // load wgpy python package corresponding to the backend.
    // if wgpy is not initialized, wgpy (and cupy) is not available.
    await pyodide.loadPackage(`../../dist/wgpy_${initWorkerResult.backend}-1.0.0-py3-none-any.whl`);
  }

  log('Loading pyodide succeeded');
  const pythonCode = await loadPythonCode();

  self.pythonIO = {
    config,
    trainingProgress: (progress) => {
      postMessage({ namespace: 'app', method: 'trainingProgress', progress });
    },
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
      start(ev.data.config).catch((reason) => log(`Worker error: ${reason}`));
      break;
  }
});
