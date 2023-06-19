function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

async function run() {
  const backend = document.querySelector('input[name="backend"]:checked').value;
  console.log(`backend: ${backend}`);
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

  // example: http://locahost:8000/?test_path=/test_chainer.py
  const testPath = (new URLSearchParams(location.search)).get('test_path') || '';
  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', backend, testPath });
}

window.addEventListener('load', () => {
  const defaultBackend = (new URLSearchParams(location.search)).get('backend') || 'webgl';
  const radioButton = document.querySelector(`input[name="backend"][value="${defaultBackend}"]`);
  if (radioButton) {
    radioButton.checked = true;
  }

  document.getElementById('run').onclick = () => {
    run().catch((error) => {
      log(`Main thread error: ${error.message}`);
    });
  };
});
