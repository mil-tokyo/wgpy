import { WgpyBackend } from './backend';
import { ComputeContextGL } from './webgl/webglComputeContext';
import { ComputeContextGPU } from './webgpu/webgpuComputeContext';

export interface WgpyInitOptions {
  // specify the order of backend to try. default: ['webgpu', 'webgl']
  // if ['webgpu', 'webgl'] is specified, webgpu will be tried first, and if it fails, webgl will be tried.
  backendOrder?: WgpyBackend[];
}

export interface WgpyInitResult {
  backend: WgpyBackend;
}

export async function initMain(worker: Worker, options: WgpyInitOptions): Promise<WgpyInitResult> {
  let contextGL: ComputeContextGL | null = null;
  let contextGPU: ComputeContextGPU | null = null;
  let initializedBackend: WgpyBackend | null = null;
  for (const backend of options.backendOrder ?? ['webgpu', 'webgl']) {
    if (backend === 'webgl') {
      contextGL = new ComputeContextGL();
      try {
        await contextGL.init();
        initializedBackend = backend;
      } catch (error) {
        console.error(
          `wgpy: failed to initialize WebGL context: ${(error as any)?.message}`
        );
        contextGL = null;
      }
    } else if (backend === 'webgpu') {
      contextGPU = new ComputeContextGPU();
      try {
        await contextGPU.init();
        initializedBackend = backend;
      } catch (error) {
        console.error(
          `wgpy: failed to initialize WebGPU context: ${(error as any)?.message}`
        );
        contextGPU = null;
      }
    } else {
      throw new Error(`wgpy: unknown backend: ${backend}`);
    }
    if (initializedBackend) {
      break;
    }
  }

  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'wgpy') {
      return;
    }
    if (e.data.method === 'init') {
      // if no backend is initialized, send initComplete with gl=gpu=null, which causes initPromiseReject
      worker.postMessage({
        namespace: 'wgpy',
        method: 'initComplete',
        gl: contextGL ? contextGL.getDeviceInfo() : null, // TODO: send device features
        gpu: contextGPU ? {} : null,
      });
    } else if (e.data.method.startsWith('gl.')) {
      if (contextGL) {
        try {
          contextGL.handleMessage(e.data, worker);
        } catch (error) {
          console.error(error);
        }
      } else {
        console.error('WebGL context is not initialized. You may have loaded wrong wgpy python package.');
      }
    } else if (e.data.method.startsWith('gpu.')) {
      if (contextGPU) {
        try {
          contextGPU.handleMessage(e.data, worker);
        } catch (error) {
          console.error(error);
        }
      } else {
        console.error('WebGPU context is not initialized. You may have loaded wrong wgpy python package.');
      }
    }
  });

  if (!initializedBackend) {
    throw new Error('wgpy: failed to initialize any backend');
  }

  return {backend: initializedBackend};
}
