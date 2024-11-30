function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

function getConfig() {
  let backend = document.querySelector('input[name="backend"]:checked').value;
  const use_gpu = backend !== 'cpu';
  if (!use_gpu) {
    backend = 'webgl';
  }
  const kernel_type = document.querySelector('input[name="kerneltype"]:checked').value;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const grid = Number(document.querySelector('input[name="grid"]:checked').value);

  return { backend, use_gpu, kernel_type, mode, grid };
}

async function run() {
  const config = getConfig();
  const worker = new Worker('worker.js');

  log('Initializing wgpy main-thread-side javascript interface');
  let initializedBackend = 'cpu';
  try {
    const initResult = await wgpy.initMain(worker, { backendOrder: [config.backend] });
    initializedBackend = initResult.backend;
  } catch (e) {
  }
  config.backend = initializedBackend; // actually initialized backend
  log(`Initialized backend: ${initializedBackend}`);

  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'app') {
      // message for library
      return;
    }

    switch (e.data.method) {
      case 'log':
        log(e.data.message);
        break;
      case 'displayImage':
        const img = document.createElement('img');
        img.src = e.data.url;
        document.getElementById('resultImage').appendChild(img);
        break;
    }
  });

  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', config });
}

window.addEventListener('load', () => {
  document.getElementById('run').onclick = () => {
    document.getElementById('run').disabled = true;
    run().catch((error) => {
      log(`Main thread error: ${error.message}`);
    });
  };
});
