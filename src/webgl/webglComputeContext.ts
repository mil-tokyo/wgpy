import { nonNull } from '../util';
import { WorkGroupDim } from '../webgpu/webgpuComputeContext';
import {
  getNNWebGLContext,
  initializeNNWebGLContext,
  TensorTextureShape,
  WebGLTensorBuffer,
  WebGLUniformItem,
} from './webglContext';

export interface GLKernelRunDescriptor {
  name: string;
  inputs: { name: string; id: number }[];
  output: number;
  uniforms: WebGLUniformItem[];
}

export interface GPUKernelRunDescriptor {
  name: string;
  tensors: number[];
  uniforms: WebGLUniformItem[];
  workGroups: { [key in WorkGroupDim]: number };
}

export interface ComputeContextGLMessageCreateBuffer {
  method: 'gl.createBuffer';
  id: number;
  textureShape: TensorTextureShape;
}

export interface ComputeContextGLMessageDisposeBuffer {
  method: 'gl.disposeBuffer';
  id: number;
}

export interface ComputeContextGLMessageSetData {
  method: 'gl.setData';
  id: number;
  data: Float32Array;
}

export interface ComputeContextGLMessageGetData {
  method: 'gl.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
  ctorType: string;
}

export interface ComputeContextGLMessageAddKernel {
  method: 'gl.addKernel';
  name: string;
  descriptor: { source: string };
}

export interface ComputeContextGLMessageRunKernel {
  method: 'gl.runKernel';
  descriptor: GLKernelRunDescriptor;
}

export type ComputeContextGLMessage =
  | ComputeContextGLMessageAddKernel
  | ComputeContextGLMessageCreateBuffer
  | ComputeContextGLMessageDisposeBuffer
  | ComputeContextGLMessageGetData
  | ComputeContextGLMessageRunKernel
  | ComputeContextGLMessageSetData;

export class ComputeContextGL {
  tensorBuffers: Map<number, WebGLTensorBuffer> = new Map();
  async init() {
    await initializeNNWebGLContext();
  }

  getDeviceInfo() {
    const ctx = getNNWebGLContext();
    return {
      maxTextureSize: ctx.maxTextureSize,
      supportsTexture32bit: ctx.supportsTexture32bit,
      supportsTexture16bit: ctx.supportsTexture16bit,
      canReadRedTexture: ctx.canReadRedTexture,
      canReadNon32bitTexture: ctx.canReadNon32bitTexture,
    };
  }

  createBuffer(id: number, textureShape: TensorTextureShape) {
    const tensorBuffer = new WebGLTensorBuffer(textureShape);
    this.tensorBuffers.set(id, tensorBuffer);
  }

  disposeBuffer(id: number) {
    const tb = this.tensorBuffers.get(id);
    if (tb) {
      tb.dispose();
      this.tensorBuffers.delete(id);
    }
  }

  setData(id: number, data: ArrayBufferView): void {
    // no pack
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return;
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint16Array> {
    // no pack
    // not necessarily async, but matching WebGPU API
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject();
    }
    // TODO consider data format
    // const data = tb.getDataRawFloat32();
    const data = tb.getDataRaw();
    return Promise.resolve(data.buffer as Uint16Array);
  }

  addKernel(name: string, descriptor: { source: string }) {
    const ctx = getNNWebGLContext();
    ctx.addKernel(name, descriptor.source);
  }

  runKernel(descriptor: GLKernelRunDescriptor) {
    const ctx = getNNWebGLContext();
    const inputs = descriptor.inputs.map(({ name, id }) => ({
      name,
      buffer: nonNull(this.tensorBuffers.get(id)),
    }));
    const output = nonNull(this.tensorBuffers.get(descriptor.output));
    ctx.runKernel(descriptor.name, inputs, output, descriptor.uniforms);
  }

  mdata: SharedArrayBuffer | null = null;
  mnotify: Int32Array | null = null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  handleMessage(message: ComputeContextGLMessage, worker: Worker) {
    switch (message.method) {
      case 'gl.addKernel':
        this.addKernel(message.name, message.descriptor);
        break;
      case 'gl.createBuffer':
        this.createBuffer(message.id, message.textureShape);
        break;
      case 'gl.disposeBuffer':
        this.disposeBuffer(message.id);
        break;
      case 'gl.getData':
        if (message.data) {
          this.mdata = message.data;
        }
        if (message.notify) {
          this.mnotify = new Int32Array(message.notify);
        }
        this.getData(message.id)
          .then((data) => {
            const ctor = {
              Float32Array: Float32Array,
              Int32Array: Int32Array,
              Uint16Array: Uint16Array,
              Uint8Array: Uint8Array,
            }[message.ctorType];
            if (!ctor) {
              throw new Error('unknown ctor type');
            }
            new ctor(this.mdata!).set(data);
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
          });
        break;
      case 'gl.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gl.setData':
        this.setData(message.id, message.data);
        break;
    }
  }
}
