function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

function getConfig() {
  const deviceElem = document.querySelector('input[name=device]:checked');
  const device = deviceElem?.value || 'cpu';
  const modelElem = document.querySelector('input[name=model]:checked');
  const model = modelElem?.value || 'MLP';
  const batchSize = Number(document.querySelector('input[name=batchSize]').value);
  return { device, model, batchSize };
}

async function run() {
  const config = getConfig();
  const worker = new Worker('worker.js');

  log('Initializing wgpy main-thread-side javascript interface');
  await wgpy.initMain(worker, {gl: true, gpu: false});

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
