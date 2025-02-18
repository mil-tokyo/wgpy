function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

const grid = 1024;

function getConfig() {
  let backend = document.querySelector('input[name="backend"]:checked').value;
  const use_gpu = backend !== 'cpu';
  if (!use_gpu) {
    backend = 'webgl';
  }
  const kernel_type = 'custom';

  return { backend, use_gpu, kernel_type, grid };
}

let worker;
async function run() {
  const config = getConfig();
  worker = new Worker('worker.js');

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
        document.getElementById('mandelbrotImage').src = e.data.url;
        break;
    }
  });

  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', config });
}

function render(x_min, x_max, y_min, y_max) {
  if (worker) {
    console.log({ namespace: 'app', method: 'render', grid, x_min, x_max, y_min, y_max })
    worker.postMessage({ namespace: 'app', method: 'render', grid, x_min, x_max, y_min, y_max });
  }
}

function debounce(func, delay) {
  let timerId;

  return function(...args) {
    clearTimeout(timerId);
    timerId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}

const debouncedRender = debounce(render, 300);

window.addEventListener('load', () => {
  document.getElementById('run').onclick = () => {
    document.getElementById('run').disabled = true;
    run().catch((error) => {
      log(`Main thread error: ${error.message}`);
    });
  };

  if (navigator.gpu) {
    document.querySelector('input[name="backend"][value="webgpu"]').checked = true;
  } else {
    document.querySelector('input[name="backend"][value="webgl"]').checked = true;
  }

    let x_min = -2.0, x_max = 0.5, y_min = -1.2, y_max = 1.2;
    const img = document.getElementById("mandelbrotImage");
  
    //render(x_min, x_max, y_min, y_max);
  
    let isDragging = false;
    let startX, startY;
    let startXMin, startXMax, startYMin, startYMax;
  
    img.addEventListener("mousedown", (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      startXMin = x_min;
      startXMax = x_max;
      startYMin = y_min;
      startYMax = y_max;
    });
  
    img.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
  
      const dx = (e.clientX - startX) / img.width * (x_max - x_min);
      const dy = (e.clientY - startY) / img.height * (y_max - y_min);
  
      x_min = startXMin - dx;
      x_max = startXMax - dx;
      y_min = startYMin - dy;
      y_max = startYMax - dy;
  
    });
  
    img.addEventListener("mouseup", () => {
      isDragging = false;
      debouncedRender(x_min, x_max, y_min, y_max);
    });
  
    img.addEventListener("mouseleave", () => {
      isDragging = false;
      debouncedRender(x_min, x_max, y_min, y_max);
    });
  
    img.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = 1.1;
      const scale = e.deltaY > 0 ? zoomFactor : 1 / zoomFactor;
  
      const mouseX = e.offsetX / img.width;
      const mouseY = e.offsetY / img.height;
  
      const x_center = x_min + mouseX * (x_max - x_min);
      const y_center = y_min + (1 - mouseY) * (y_max - y_min);
  
      const new_width = (x_max - x_min) * scale;
      const new_height = (y_max - y_min) * scale;
  
      x_min = x_center - mouseX * new_width;
      x_max = x_min + new_width;
      y_min = y_center - (1 - mouseY) * new_height;
      y_max = y_min + new_height;
  
      debouncedRender(x_min, x_max, y_min, y_max);
    });
  
    document.getElementById('resetZoom').addEventListener("click", (e) => {
      e.preventDefault();
      x_min = -2.0, x_max = 0.5, y_min = -1.2, y_max = 1.2;
      debouncedRender(x_min, x_max, y_min, y_max);
    });
});
