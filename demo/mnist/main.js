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
  const modelElem = document.querySelector('input[name=model]:checked');
  const model = modelElem?.value || 'MLP';
  const batchSize = Number(document.querySelector('input[name=batchSize]').value);
  return { backendOrder, model, batchSize };
}

function displayTrainingProgress(progress) {
  document.getElementById('iterations').innerText = progress.iterations.toString();
  document.getElementById('trainingLoss').innerText = progress.trainingLoss.toFixed(3);
  document.getElementById('testLoss').innerText = progress.testLoss.toFixed(3);
  document.getElementById('testAccuracy').innerText = `${(progress.testAccuracy * 100).toFixed(0)} %`;
  for (let i = 0; i < progress.predictedLabels.length; i++) {
    const td = document.getElementById(`prediction${i}`);
    td.innerText = progress.predictedLabels[i].toString();
    td.className = progress.predictedLabels[i] === progress.correctLabels[i] ? 'correct' : 'wrong';
  }
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
      case 'trainingProgress':
        displayTrainingProgress(JSON.parse(e.data.progress));
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
