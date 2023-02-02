import { GLKernelRunDescriptor } from './webgl/webglComputeContext';
import { TensorTextureShape } from './webgl/webglContext';
import { GPUKernelRunDescriptor } from './webgpu/webgpuComputeContext';

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

      const dataSrc = new ctor(
        commBufUint8Array!.buffer,
        commBufUint8Array!.byteOffset,
        size
      );
      const transferData = new ctor(size);
      transferData.set(dataSrc);
      postToMain({ method: 'gl.setData', id, data: transferData }, [
        transferData.buffer,
      ]);
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
          `buffer size insufficient: ${
            size * ctor.BYTES_PER_ELEMENT
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
      const dataSrc = new ctor(
        commBufUint8Array!.buffer,
        commBufUint8Array!.byteOffset,
        size
      );
      dataSrc.set(placeholderData);
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
      forWriteFromCPU: boolean,
      forReadToCPU: boolean
    ) => {
      postToMain({
        method: 'gpu.createBuffer',
        id,
        byteLength,
        forWriteFromCPU,
        forReadToCPU,
      });
    },
    disposeBuffer: (id: number) => {
      postToMain({ method: 'gpu.disposeBuffer', id });
    },
    setData: (id: number, data: any) => {
      const buffer = data.getBuffer();
      data.destroy();
      try {
        const len = buffer.nbytes / buffer.itemsize;
        const transferData = new Float32Array(len);
        transferData.set(buffer.data);
        postToMain({ method: 'gpu.setData', id, data: transferData }, [
          transferData.buffer,
        ]);
      } finally {
        buffer.release();
      }
    },
    getData: (id: number, data: any) => {
      const buffer = data.getBuffer();
      data.destroy();
      try {
        const notifyBuffer = new SharedArrayBuffer(4);
        const notifyBufferView = new Int32Array(notifyBuffer);
        notifyBufferView[0] = 0;
        const placeholderBuffer = new SharedArrayBuffer(buffer.nbytes);
        const placeholderData = new Float32Array(placeholderBuffer);
        postToMain({
          method: 'gpu.getData',
          id,
          data: placeholderData,
          notify: notifyBufferView,
        });
        Atomics.wait(notifyBufferView, 0, 0);
        buffer.data.set(placeholderData);
      } finally {
        buffer.release();
      }
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

export async function initWorker() {
  let initPromiseResolve: () => void = () => {
    throw new Error('unexpected call of initPromiseResolve');
  };
  addEventListener('message', (e) => {
    if (e.data.namespace !== 'wgpy') {
      return;
    }
    switch (e.data.method) {
      case 'initComplete':
        if (e.data.gl != null) {
          initGLInterface(true, e.data.gl);
        } else {
          initGLInterface(false, null);
        }
        if (e.data.gpu != null) {
          initGPUInterface(true, e.data.gpu);
        } else {
          initGPUInterface(false, null);
        }
        initPromiseResolve();
        break;
    }
  });

  postToMain({ method: 'init' });

  return new Promise<void>((resolve) => {
    initPromiseResolve = resolve;
  });
}
