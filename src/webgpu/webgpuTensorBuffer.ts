import { DType, TypedArrayConstructor } from '../dtype';
import { getNNWebGPUContext } from './webgpuContext';

let webgpuAllocCount = 0;
export const existingBuffers: Set<WebGPUTensorBuffer> = new Set();

export type TypedArrayTypesForWebGPUBuffer = Float32Array | Int32Array | Uint32Array;

export interface WebGPUBufferShape {
  byteLength: number;
  forWriteFromCPU: boolean;
  forReadToCPU: boolean;
}

export class WebGPUTensorBuffer {
  public ref: number;
  gpuBuffer: GPUBuffer;

  private mappedForWriteFromCPU: boolean;

  constructor(public readonly bufferShape: WebGPUBufferShape) {
    const ctx = getNNWebGPUContext();
    this.ref = 1;
    let usage = GPUBufferUsage.STORAGE;
    if (bufferShape.forReadToCPU) {
      usage |= GPUBufferUsage.COPY_SRC;
    }
    this.gpuBuffer = ctx.device.createBuffer({
      mappedAtCreation: bufferShape.forWriteFromCPU,
      size: bufferShape.byteLength,
      usage,
    });
    this.mappedForWriteFromCPU = bufferShape.forWriteFromCPU;
    webgpuAllocCount++;
    existingBuffers.add(this);
  }

  setDataRaw(data: TypedArrayTypesForWebGPUBuffer): void {
    if (!this.mappedForWriteFromCPU) {
      // TODO: enable write again by creating temporary buffer and copybuffertobuffer
      throw new Error(
        'The buffer is not mapped. WebGPUTensor can only be written just after creation.'
      );
    }

    const ab = this.gpuBuffer.getMappedRange();
    // create same typedarray as data
    const mappedArray = new (data.constructor as TypedArrayConstructor)(ab);
    mappedArray.set(data);
    this.gpuBuffer.unmap();
    this.mappedForWriteFromCPU = false;
  }

  async getDataRaw(dtype: DType): Promise<TypedArrayTypesForWebGPUBuffer> {
    if (!this.bufferShape.forReadToCPU) {
      throw new Error(
        'forReadToCPU flag is not set for this WebGPUTensor. Please use WebGPUTensor.copy() to create readable tensor.'
      );
    }
    const ctx = getNNWebGPUContext();
    let ctor: typeof Float32Array | typeof Int32Array | typeof Uint32Array;
    let itemCount: number;
    switch (dtype) {
      case 'float32':
        ctor = Float32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'int32':
        ctor = Int32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      case 'uint8':
      case 'bool':
        // Stored in Uint32Array instead of Uint8Array
        ctor = Uint32Array;
        itemCount = this.bufferShape.byteLength / ctor.BYTES_PER_ELEMENT;
        break;
      default:
        throw new Error(`Unknown dtype ${dtype}`);
    }

    const data: TypedArrayTypesForWebGPUBuffer = new ctor(itemCount),
      dst = ctx.device.createBuffer({
        size: this.bufferShape.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      commandEncoder = ctx.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.gpuBuffer,
      0,
      dst,
      0,
      this.bufferShape.byteLength
    );
    ctx.device.queue.submit([commandEncoder.finish()]);
    await dst.mapAsync(GPUMapMode.READ);
    const arrayBuffer = dst.getMappedRange(),
      buffer_mapped_array = new ctor(arrayBuffer, 0, itemCount);
    data.set(buffer_mapped_array);
    dst.unmap();
    dst.destroy();
    return data;
  }

  dispose() {
    this.gpuBuffer.destroy();
    webgpuAllocCount--;
    existingBuffers.delete(this);
    (this as { gpuBuffer: GPUBuffer | null }).gpuBuffer = null;
  }
}
