import { getNNWebGPUContext } from './webgpuContext';

let webgpuAllocCount = 0;
export const existingBuffers: Set<WebGPUTensorBuffer> = new Set();

export interface WebGPUBufferShape {
  byteLength: number;
}

export class WebGPUTensorBuffer {
  gpuBuffer: GPUBuffer;

  // private mappedForWriteFromCPU: boolean;

  constructor(public readonly bufferShape: WebGPUBufferShape) {
    const ctx = getNNWebGPUContext();
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    // if (bufferShape.forReadToCPU) {
    //   usage |= GPUBufferUsage.COPY_SRC;
    // }
    this.gpuBuffer = ctx.device.createBuffer({
      mappedAtCreation: false, //bufferShape.forWriteFromCPU,
      size: bufferShape.byteLength,
      usage,
    });
    // this.mappedForWriteFromCPU = bufferShape.forWriteFromCPU;
    webgpuAllocCount++;
    existingBuffers.add(this);
  }

  setDataRaw(data: Uint8Array): void {
    // 新しいバッファに書いて、copyBufferToBufferで明示的にコピーする仕様に変える
    if (!this.mappedForWriteFromCPU) {
      // TODO: enable write again by creating temporary buffer and copybuffertobuffer
      throw new Error(
        'The buffer is not mapped. WebGPUTensor can only be written just after creation.'
      );
    }

    const ab = this.gpuBuffer.getMappedRange();
    // create same typedarray as data
    const mappedArray = new Uint8Array(ab);
    mappedArray.set(data);
    this.gpuBuffer.unmap();
    this.mappedForWriteFromCPU = false;
  }

  async getDataRaw(): Promise<Uint8Array> {
    if (!this.bufferShape.forReadToCPU) {
      throw new Error(
        'forReadToCPU flag is not set for this WebGPUTensor. Please use WebGPUTensor.copy() to create readable tensor.'
      );
    }
    const ctx = getNNWebGPUContext();

    const data = new Uint8Array(this.bufferShape.byteLength),
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
      buffer_mapped_array = new Uint8Array(arrayBuffer, 0, this.bufferShape.byteLength);
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
