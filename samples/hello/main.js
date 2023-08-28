function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

async function run() {
  const inputData = JSON.parse(document.getElementById('inputData').value);
  log(`This program caluculates ${JSON.stringify(inputData)} * 2 in python`);
  const backend = document.querySelector('input[name="backend"]:checked').value;
  console.log(`backend: ${backend}`);
  const worker = new Worker('worker.js');

  log('Initializing wgpy main-thread-side javascript interface');
  const initResult = await wgpy.initMain(worker, {backendOrder: [backend]});
  log(`initResult: ${JSON.stringify(initResult)}`);

  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'app') {
      // message for library
      return;
    }

    switch (e.data.method) {
      case 'log':
        log(e.data.message);
        break;
      case 'output':
        log(`Output: ${JSON.stringify(e.data.data)}`);
        break;
    }
  });

  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', backend, data: inputData });
}

window.addEventListener('load', () => {
  document.getElementById('run').onclick = () => {
    run().catch((error) => {
      log(`Main thread error: ${error.message}`);
    });
  };
});
