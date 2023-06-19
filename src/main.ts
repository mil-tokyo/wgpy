import { ComputeContextGL } from './webgl/webglComputeContext';
import { ComputeContextGPU } from './webgpu/webgpuComputeContext';

export interface WgpyInitOptions {
  gl?: boolean;
  gpu?: boolean;
}

export async function initMain(worker: Worker, options: WgpyInitOptions) {
  let contextGL: ComputeContextGL | null = null;
  let contextGPU: ComputeContextGPU | null = null;
  if (options.gl) {
    contextGL = new ComputeContextGL();
    try {
      await contextGL.init();
    } catch (error) {
      console.error(
        `wgpy: failed to initialize WebGL context: ${(error as any)?.message}`
      );
      contextGL = null;
    }
  }
  if (options.gpu) {
    contextGPU = new ComputeContextGPU();
    try {
      await contextGPU.init();
    } catch (error) {
      console.error(
        `wgpy: failed to initialize WebGPU context: ${(error as any)?.message}`
      );
      contextGPU = null;
    }
  }
  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'wgpy') {
      return;
    }
    if (e.data.method === 'init') {
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
        console.error('WebGL context is not initialized. Call initMain with options={gl: true}.');
      }
    } else if (e.data.method.startsWith('gpu.')) {
      if (contextGPU) {
        try {
          contextGPU.handleMessage(e.data, worker);
        } catch (error) {
          console.error(error);
        }
      } else {
        console.error('WebGPU context is not initialized. Call initMain with options={gpu: true}.');
      }
    }
  });
}
