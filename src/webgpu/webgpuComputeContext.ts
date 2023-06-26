import { nonNull } from '../util';
import { getNNWebGPUContext, initializeNNWebGPUContext } from './webgpuContext';
import {
  WebGPUTensorBuffer,
} from './webgpuTensorBuffer';

export type WorkGroupDim = 'x' | 'y' | 'z';

export interface GPUKernelRunDescriptor {
  name: string;
  tensors: number[];
  workGroups: { [key in WorkGroupDim]: number };
}

export interface ComputeContextGPUMessageCreateBuffer {
  method: 'gpu.createBuffer';
  id: number;
  byteLength: number;
}

export interface ComputeContextGPUMessageDisposeBuffer {
  method: 'gpu.disposeBuffer';
  id: number;
}

export interface ComputeContextGPUMessageSetData {
  method: 'gpu.setData';
  id: number;
  data: Uint8Array;
}

export interface ComputeContextGPUMessageGetData {
  method: 'gpu.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
}

export interface ComputeContextGPUMessageAddKernel {
  method: 'gpu.addKernel';
  name: string;
  descriptor: { source: string; bindingTypes: GPUBufferBindingType[] };
}

export interface ComputeContextGPUMessageRunKernel {
  method: 'gpu.runKernel';
  descriptor: GPUKernelRunDescriptor;
}

export type ComputeContextGPUMessage =
  | ComputeContextGPUMessageAddKernel
  | ComputeContextGPUMessageCreateBuffer
  | ComputeContextGPUMessageDisposeBuffer
  | ComputeContextGPUMessageGetData
  | ComputeContextGPUMessageRunKernel
  | ComputeContextGPUMessageSetData;

export class ComputeContextGPU {
  tensorBuffers: Map<number, WebGPUTensorBuffer> = new Map();
  async init() {
    await initializeNNWebGPUContext();
  }

  createBuffer(
    id: number,
    byteLength: number,
  ) {
    const tensorBuffer = new WebGPUTensorBuffer({
      byteLength,
    });
    this.tensorBuffers.set(id, tensorBuffer);
  }

  disposeBuffer(id: number) {
    const tb = this.tensorBuffers.get(id);
    if (tb) {
      tb.dispose();
      this.tensorBuffers.delete(id);
    }
  }

  setData(id: number, data: Uint8Array): void {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return;
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint8Array> {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject();
    }
    return tb.getDataRaw() as Promise<Uint8Array>;
  }

  addKernel(
    name: string,
    descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
  ) {
    const ctx = getNNWebGPUContext();
    ctx.createPipeline(name, descriptor.source, descriptor.bindingTypes);
  }

  runKernel(descriptor: GPUKernelRunDescriptor) {
    const ctx = getNNWebGPUContext();
    const tensor = descriptor.tensors.map((id) =>
      nonNull(this.tensorBuffers.get(id))
    );
    ctx.runKernel({
      pipelineName: descriptor.name,
      tensorBuffers: tensor,
      workGroups: descriptor.workGroups,
    });
  }

  mdata: SharedArrayBuffer | null = null;
  mnotify: Int32Array | null = null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  handleMessage(message: ComputeContextGPUMessage, worker: Worker) {
    switch (message.method) {
      case 'gpu.addKernel':
        this.addKernel(message.name, message.descriptor);
        break;
      case 'gpu.createBuffer':
        this.createBuffer(
          message.id,
          message.byteLength,
        );
        break;
      case 'gpu.disposeBuffer':
        this.disposeBuffer(message.id);
        break;
      case 'gpu.getData':
        if (message.data) {
          this.mdata = message.data;
        }
        if (message.notify) {
          this.mnotify = new Int32Array(message.notify);
        }
        this.getData(message.id)
          .then((data) => {
            (new Uint8Array(this.mdata!)).set(data);
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
          });
        break;
      case 'gpu.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gpu.setData':
        this.setData(message.id, message.data);
        break;
    }
  }
}
