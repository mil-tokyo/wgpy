import { WgpyBackend } from './backend';
import { GLKernelRunDescriptor } from './webgl/webglComputeContext';
import { TensorTextureShape } from './webgl/webglContext';
import { GPUKernelRunDescriptor } from './webgpu/webgpuComputeContext';

export interface WgpyInitWorkerResult {
  backend: WgpyBackend;
}

function dictToObj(dict: any) {
  // Convert python dict to JS object. func({"x":1,"y":[2,3]}) in python
  return dict.toJs({
    dict_converter: Object.fromEntries,
    create_proxies: false,
  });
}

function postToMain(obj: any, transfer: Transferable[] = []) {
  postMessage({ namespace: 'wgpy', ...obj }, transfer);
}

function initGLInterface(glAvailable: boolean, glDeviceInfo: any) {
  let sharedBufferSent = false;
  let notifyBuffer: SharedArrayBuffer | undefined = undefined;
  let notifyBufferView: Int32Array | undefined = undefined;
  let placeholderBuffer: SharedArrayBuffer | undefined = undefined;
  let commBuf: any = undefined;
  let commBufUint8Array: Uint8Array | undefined = undefined;
  (globalThis as any).gl = {
    isAvailable: () => {
      return glAvailable;
    },
    getDeviceInfo: () => {
      return glDeviceInfo;
    },
    createBuffer: (id: number, textureShape: TensorTextureShape) => {
      postToMain({
        method: 'gl.createBuffer',
        id,
        textureShape: dictToObj(textureShape),
      });
    },
    disposeBuffer: (id: number) => {
      postToMain({ method: 'gl.disposeBuffer', id });
    },
    setCommBuf: (data: any) => {
      if (commBuf) {
        commBuf.release();
      }
      commBuf = data.getBuffer();
      // data.destroy() takes relatively long time, so use the same buffer for every setData / getData.
      data.destroy();
      commBufUint8Array = commBuf.data;
    },
    setData: (id: number, ctorType: string, size: number) => {
      const ctor = {
        Float32Array: Float32Array,
        Int32Array: Int32Array,
        Uint16Array: Uint16Array,
        Uint8Array: Uint8Array,
      }[ctorType];
      if (!ctor) {
        throw new Error('ctorType unknown ' + ctorType);
      }

      let dataSrc: Float32Array | Int32Array | Uint16Array | Uint8Array;
      try {
        // same as setData
        dataSrc = new ctor(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          size
        );
      } catch (e) {
        return false;
      }
      const transferData = new ctor(size);
      transferData.set(dataSrc);
      postToMain({ method: 'gl.setData', id, data: transferData }, [
        transferData.buffer,
      ]);
      return true;
    },
    getData: (id: number, ctorType: string, size: number) => {
      const ctor = {
        Float32Array: Float32Array,
        Int32Array: Int32Array,
        Uint16Array: Uint16Array,
        Uint8Array: Uint8Array,
      }[ctorType];
      if (!ctor) {
        throw new Error('ctorType unknown ' + ctorType);
      }
      let dataSrc: Float32Array | Int32Array | Uint16Array | Uint8Array;
      try {
        // same as setData
        dataSrc = new ctor(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          size
        );
      } catch (e) {
        return false;
      }
      if (!notifyBuffer) {
        notifyBuffer = new SharedArrayBuffer(4);
        notifyBufferView = new Int32Array(notifyBuffer);
        notifyBufferView[0] = 0;
      }
      if (!placeholderBuffer) {
        placeholderBuffer = new SharedArrayBuffer(16 * 1024 * 1024 * 4);
      }

      if (size * ctor.BYTES_PER_ELEMENT > placeholderBuffer.byteLength) {
        throw new Error(
          `buffer size insufficient: ${size * ctor.BYTES_PER_ELEMENT
          } bytes required`
        );
      }
      notifyBufferView![0] = 0;
      if (sharedBufferSent) {
        postToMain({ method: 'gl.getData', id, ctorType });
      } else {
        postToMain({
          method: 'gl.getData',
          id,
          data: placeholderBuffer,
          notify: notifyBuffer,
          ctorType,
        });
        sharedBufferSent = true;
      }
      // if buffer[0] = 1 is written before Atomics.wait, it does not wait.
      Atomics.wait(notifyBufferView!, 0, 0);

      const placeholderData = new ctor(placeholderBuffer, 0, size);
      dataSrc.set(placeholderData);
      return true;
    },
    addKernel: (name: string, descriptor: { source: string }) => {
      postToMain({
        method: 'gl.addKernel',
        name,
        descriptor: dictToObj(descriptor),
      });
    },
    runKernel: (descriptor: GLKernelRunDescriptor) => {
      postToMain({ method: 'gl.runKernel', descriptor: dictToObj(descriptor) });
    },
  };
}

function initGPUInterface(gpuAvailable: boolean, gpuDeviceInfo: any) {
  let sharedBufferSent = false;
  let notifyBuffer: SharedArrayBuffer | undefined = undefined;
  let notifyBufferView: Int32Array | undefined = undefined;
  let placeholderBuffer: SharedArrayBuffer | undefined = undefined;
  let commBuf: any = undefined;
  let commBufUint8Array: Uint8Array | undefined = undefined;
  (globalThis as any).gpu = {
    isAvailable: () => {
      return gpuAvailable;
    },
    getDeviceInfo: () => {
      return gpuDeviceInfo;
    },
    createBuffer: (
      id: number,
      byteLength: number,
    ) => {
      postToMain({
        method: 'gpu.createBuffer',
        id,
        byteLength,
      });
    },
    createMetaBuffer: (
      id: number,
      byteLength: number,
    ) => {
      const dataSrc = new Uint8Array(
        commBufUint8Array!.buffer,
        commBufUint8Array!.byteOffset,
        byteLength
      );
      const transferData = new Uint8Array(byteLength);
      transferData.set(dataSrc);
      postToMain({
        method: 'gpu.createMetaBuffer',
        id,
        byteLength,
        data: transferData
      }, [transferData.buffer]);
    },
    disposeBuffer: (id: number) => {
      postToMain({ method: 'gpu.disposeBuffer', id });
    },
    setCommBuf: (data: any) => {
      if (commBuf) {
        commBuf.release();
      }
      commBuf = data.getBuffer();
      // data.destroy() takes relatively long time, so use the same buffer for every setData / getData.
      data.destroy();
      commBufUint8Array = commBuf.data;
    },
    setData: (id: number, byteLength: number) => {
      // When wasm buffer is reallocated, commBufUint8Array is detached.
      // 'TypeError: Cannot perform Construct on a detached ArrayBuffer' is thrown.
      let dataSrc: Uint8Array;
      try {
        dataSrc = new Uint8Array(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          byteLength
        );
      } catch (e) {
        return false;
      }
      const transferData = new Uint8Array(byteLength);
      transferData.set(dataSrc);
      postToMain({ method: 'gpu.setData', id, data: transferData }, [
        transferData.buffer,
      ]);
      return true;
    },
    getData: (id: number, byteLength: number) => {
      let dataSrc: Uint8Array;
      try {
        // same as setData
        dataSrc = new Uint8Array(
          commBufUint8Array!.buffer,
          commBufUint8Array!.byteOffset,
          byteLength
        );
      } catch (e) {
        return false;
      }
      if (!notifyBuffer) {
        notifyBuffer = new SharedArrayBuffer(4);
        notifyBufferView = new Int32Array(notifyBuffer);
        notifyBufferView[0] = 0;
      }
      if (!placeholderBuffer) {
        placeholderBuffer = new SharedArrayBuffer(16 * 1024 * 1024 * 4);
      }

      if (byteLength > placeholderBuffer.byteLength) {
        throw new Error(
          `buffer size insufficient: ${byteLength
          } bytes required`
        );
      }
      notifyBufferView![0] = 0;
      if (sharedBufferSent) {
        postToMain({ method: 'gpu.getData', id });
      } else {
        postToMain({
          method: 'gpu.getData',
          id,
          data: placeholderBuffer,
          notify: notifyBuffer,
        });
        sharedBufferSent = true;
      }

      // if buffer[0] = 1 is written before Atomics.wait, it does not wait.
      Atomics.wait(notifyBufferView!, 0, 0);

      const placeholderData = new Uint8Array(placeholderBuffer, 0, byteLength);
      dataSrc.set(placeholderData);
      return true;
    },
    addKernel: (
      name: string,
      descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
    ) => {
      postToMain({
        method: 'gpu.addKernel',
        name,
        descriptor: dictToObj(descriptor),
      });
    },
    runKernel: (descriptor: GPUKernelRunDescriptor) => {
      postToMain({
        method: 'gpu.runKernel',
        descriptor: dictToObj(descriptor),
      });
    },
  };
}

export async function initWorker(): Promise<WgpyInitWorkerResult> {
  let initPromiseResolve: (initResult: WgpyInitWorkerResult) => void = () => {
    throw new Error('unexpected call of initPromiseResolve');
  };
  let initPromiseReject: (reason: any) => void = () => {
    throw new Error('unexpected call of initPromiseReject');
  };
  addEventListener('message', (e) => {
    if (e.data.namespace !== 'wgpy') {
      return;
    }
    switch (e.data.method) {
      case 'initComplete':
        let backend: WgpyBackend | null = null;
        if (e.data.gl != null) {
          backend = 'webgl';
          initGLInterface(true, e.data.gl);
        } else {
          initGLInterface(false, null);
        }
        if (e.data.gpu != null) {
          backend = 'webgpu';
          initGPUInterface(true, e.data.gpu);
        } else {
          initGPUInterface(false, null);
        }
        if (backend) {
          initPromiseResolve({ backend });
        } else {
          initPromiseReject(new Error('wgpy: failed to initialize any backend'));
        }
        break;
    }
  });

  postToMain({ method: 'init' });

  return new Promise<WgpyInitWorkerResult>((resolve, reject) => {
    initPromiseResolve = resolve;
    initPromiseReject = reject;
  });
}
