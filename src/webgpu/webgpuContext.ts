import { WebGPUTensorBuffer } from './webgpuTensorBuffer';

interface WebGPURunnerPipeline {
  bindGroupLayout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline;
}

type WorkGroupDim = 'x' | 'y' | 'z';

export interface WebGPUMetaBufferContentElement {
  value: number;
  type: 'int32' | 'uint32' | 'float32';
}

export interface WebGPUMetaBufferContent {
  elements: WebGPUMetaBufferContentElement[];
}

export interface WebGPURunnerRequest {
  pipelineName: string;
  tensorBuffers: WebGPUTensorBuffer[];
  workGroups: { [key in WorkGroupDim]: number };
}

export class NNWebGPUContext {
  initialized: boolean;

  isSupported: boolean;

  device!: GPUDevice;

  private pipelines: Map<string, WebGPURunnerPipeline>;

  constructor() {
    if (
      typeof navigator.gpu !== 'object' ||
      typeof navigator.gpu.requestAdapter !== 'function'
    ) {
      throw new Error('WebGPU is not supported on this browser');
    }
    this.initialized = false;
    this.isSupported = false;
    this.pipelines = new Map();
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const adapter = await navigator.gpu!.requestAdapter();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    this.device = (await adapter!.requestDevice()) as GPUDevice;
    if (!this.device) {
      throw new Error('GPUAdapter.requestDevice() returned null');
    }
    this.isSupported = true;
    this.initialized = true;
  }

  hasPipeline(name: string): boolean {
    return this.pipelines.has(name);
  }

  createPipeline(name: string, source: string, bindingTypes: GPUBufferBindingType[]): void {
    if (this.hasPipeline(name)) {
      return;
    }
    const { device } = this,
      bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < bindingTypes.length; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindingTypes[i] },
      });
    }
    const bindGroupLayout = device.createBindGroupLayout({
        entries: bindings,
      }),
      pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      shaderModule = device.createShaderModule({ code: source }),
      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

    this.pipelines.set(name, { bindGroupLayout, pipeline });
  }

  runKernel(request: WebGPURunnerRequest): void {
    const pipeline = this.pipelines.get(request.pipelineName);
    if (!pipeline) {
      throw new Error(`Pipeline ${pipeline} not found`);
    }
    const { device } = this,
      entries: GPUBindGroupEntry[] = request.tensorBuffers.map((t, i) => ({
        binding: i,
        resource: {
          buffer: t.gpuBuffer,
          size: t.bufferShape.byteLength,
        },
      }));
    const bindGroup = device.createBindGroup({
        layout: pipeline.bindGroupLayout,
        entries,
      }),
      commandEncoder = device.createCommandEncoder(),
      passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(pipeline.pipeline);
    passEncoder.dispatchWorkgroups(
      request.workGroups.x,
      request.workGroups.y,
      request.workGroups.z
    );
    if (passEncoder.end) {
      passEncoder.end();
    } else {
      // deprecated
      // Firefox Nightly 111 has this
      (passEncoder as any).endPass();
    }

    device.queue.submit([commandEncoder.finish()]);
  }
}

let context: NNWebGPUContext | null = null;
export async function initializeNNWebGPUContext(): Promise<void> {
  context = new NNWebGPUContext();
  try {
    await context.initialize();
  } catch (error) {
    context = null;
    throw error;
  }
}

export function getNNWebGPUContext(): NNWebGPUContext {
  if (!context) {
    throw new Error('WebGPU Context does not exist');
  }
  return context;
}
