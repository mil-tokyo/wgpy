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
    const ctx = getNNWebGPUContext();
    const copySrcBuffer = ctx.device.createBuffer({
      mappedAtCreation: true, // by using this option, async is not needed
      size: this.gpuBuffer.size,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
    });

    const ab = copySrcBuffer.getMappedRange();
    const mappedArray = new Uint8Array(ab);
    mappedArray.set(data);
    copySrcBuffer.unmap();

    const commandEncoder = ctx.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(copySrcBuffer, 0, this.gpuBuffer, 0, this.gpuBuffer.size);

    ctx.device.queue.submit([commandEncoder.finish()]);

    copySrcBuffer.destroy();
  }

  async getDataRaw(): Promise<Uint8Array> {
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
