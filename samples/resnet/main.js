function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

function getConfig() {
  let backendOrder = [];
  for (const backend of ['webgpu', 'webgl']) {
    if (document.getElementById(backend).checked) {
      backendOrder.push(backend);
    }
  }
  const batchSize = Number(document.querySelector('input[name=batchSize]').value);
  return { backendOrder, batchSize };
}

async function run() {
  const config = getConfig();
  const worker = new Worker('worker.js');

  log('Initializing wgpy main-thread-side javascript interface');
  let initializedBackend = 'cpu';
  try {
    const initResult = await wgpy.initMain(worker, { backendOrder: config.backendOrder });
    initializedBackend = initResult.backend;
  } catch (e) {
  }
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
