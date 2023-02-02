import { nonNull } from '../util';
import { DType } from '../dtype';

// [x y u v] * [upper-left, lower-left, upper-right, lower-right]
const vertexArray = new Float32Array([-1, +1, -1, -1, +1, +1, +1, -1]);
const vertex_shader_source_2 = `#version 300 es
precision highp float;
in vec2 _xy;
void main() { 
  gl_Position = vec4(_xy, 0, 1); 
}
`;

export interface WebGLUniformItem {
  name: string;
  value: number;
  type: 'float' | 'int';
}

export interface TensorTextureShapeFormat {
  internalFormat: number;
  format: number;
  type: number;
}

export const tensorTextureShapeFormatR32F = {
  internalFormat: WebGL2RenderingContext.R32F,
  format: WebGL2RenderingContext.RED,
  type: WebGL2RenderingContext.FLOAT,
};

export const tensorTextureShapeFormatR16F = {
  internalFormat: WebGL2RenderingContext.R16F,
  format: WebGL2RenderingContext.RED,
  type: WebGL2RenderingContext.HALF_FLOAT,
};

export const tensorTextureShapeFormatR32I = {
  internalFormat: WebGL2RenderingContext.R32I,
  format: WebGL2RenderingContext.RED_INTEGER,
  type: WebGL2RenderingContext.INT,
};

export const tensorTextureShapeFormatR8UI = {
  internalFormat: WebGL2RenderingContext.R8UI,
  format: WebGL2RenderingContext.RED_INTEGER,
  type: WebGL2RenderingContext.UNSIGNED_BYTE,
};

export const tensorTextureShapeFormatRGBA32F = {
  internalFormat: WebGL2RenderingContext.RGBA32F,
  format: WebGL2RenderingContext.RGBA,
  type: WebGL2RenderingContext.FLOAT,
};

export const tensorTextureShapeFormatRGBA16F = {
  internalFormat: WebGL2RenderingContext.RGBA16F,
  format: WebGL2RenderingContext.RGBA,
  type: WebGL2RenderingContext.HALF_FLOAT,
};

export const tensorTextureShapeFormatRGBA32I = {
  internalFormat: WebGL2RenderingContext.RGBA32I,
  format: WebGL2RenderingContext.RGBA_INTEGER,
  type: WebGL2RenderingContext.INT,
};

export const tensorTextureShapeFormatRGBA8UI = {
  internalFormat: WebGL2RenderingContext.RGBA8UI,
  format: WebGL2RenderingContext.RGBA_INTEGER,
  type: WebGL2RenderingContext.UNSIGNED_BYTE,
};

export function getTensorTextureShapeFormatForDType(
  dtype: DType,
  supportsTexture32bit?: boolean
): TensorTextureShapeFormat {
  let b32: boolean;
  if (supportsTexture32bit == null) {
    const context = getNNWebGLContext();
    b32 = context.supportsTexture32bit;
  } else {
    b32 = supportsTexture32bit;
  }
  let format: TensorTextureShapeFormat;
  switch (dtype) {
    case 'float32':
      format = b32
        ? tensorTextureShapeFormatR32F
        : tensorTextureShapeFormatR16F;
      break;
    case 'int32':
      format = tensorTextureShapeFormatR32I;
      break;
    case 'uint8':
      format = tensorTextureShapeFormatR8UI;
      break;
    case 'bool':
      format = tensorTextureShapeFormatR8UI;
      break;
    default:
      throw new Error(`WebGL texture for dtype ${dtype} is not yet supported`);
  }
  return format;
}

export const tensorTextureShapeFormatDefault = tensorTextureShapeFormatR32F;

export type TensorTextureShapeDim = '2D' | '2DArray';

export interface TensorTextureShape2D extends TensorTextureShapeFormat {
  dim: '2D';
  width: number;
  height: number;
}

export interface TensorTextureShape2DArray extends TensorTextureShapeFormat {
  dim: '2DArray';
  width: number;
  height: number;
  depth: number;
}

export type TensorTextureShape =
  | TensorTextureShape2D
  | TensorTextureShape2DArray;

export class WebGLTensorBuffer {
  public readonly texture: WebGLTexture;
  public ref: number;
  public target: number;
  private isBoundToDrawFrameBuffer = false;
  private readTextureUnitIndices: number[] = [];
  public dimPerPixel: number;
  public textureLength: number;
  constructor(public readonly textureShape: TensorTextureShape) {
    this.ref = 1;
    const ctx = getNNWebGLContext();
    this.texture = ctx.createTexture(textureShape);
    switch (textureShape.format) {
      case WebGL2RenderingContext.RED:
      case WebGL2RenderingContext.RED_INTEGER:
        this.dimPerPixel = 1;
        break;
      case WebGL2RenderingContext.RGBA:
      case WebGL2RenderingContext.RGBA_INTEGER:
        this.dimPerPixel = 4;
        break;
      default:
        throw new Error();
    }
    switch (textureShape.dim) {
      case '2D':
        this.target = WebGL2RenderingContext.TEXTURE_2D;
        this.textureLength =
          textureShape.height * textureShape.width * this.dimPerPixel;
        break;
      case '2DArray':
        this.target = WebGL2RenderingContext.TEXTURE_2D_ARRAY;
        this.textureLength =
          textureShape.depth *
          textureShape.height *
          textureShape.width *
          this.dimPerPixel;
        break;
    }
  }

  dispose() {
    const ctx = getNNWebGLContext();
    ctx.gl.deleteTexture(this.texture);
    (this as { texture: WebGLTexture | null }).texture = null;
  }

  bindToReadTexture(unit: number): void {
    if (this.texture == null) {
      throw new Error('This texture is already deleted');
    }
    if (this.isBoundToDrawFrameBuffer)
      throw Error(
        'This buffer is already registered as draw buffer. ' +
          'You may forgot to unbind the binding while previous operations.'
      );

    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(this.target, this.texture);

    this.readTextureUnitIndices.push(unit);
  }

  unbindFromReadTexture(): void {
    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    for (const unit of this.readTextureUnitIndices) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(this.target, null);
    }

    this.readTextureUnitIndices = [];
  }

  bindToDrawTexture(layer = 0): void {
    if (this.texture == null) {
      throw new Error('This texture is already deleted');
    }
    if (this.readTextureUnitIndices.length > 0)
      throw Error(
        'This buffer is already registered as read buffer. ' +
          'You cannot bind a texture as both read and draw texture buffer at same time.'
      );
    if (this.isBoundToDrawFrameBuffer)
      throw Error(
        'This buffer is already registered as draw buffer. ' +
          'You may forgot to unbind the binding while previous operations.'
      );

    const ctx = getNNWebGLContext();
    const { gl } = ctx;
    gl.viewport(0, 0, this.textureShape.width, this.textureShape.height);
    gl.scissor(0, 0, this.textureShape.width, this.textureShape.height);

    switch (this.textureShape.dim) {
      case '2D':
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          this.texture,
          0
        );
        break;
      case '2DArray':
        gl.framebufferTextureLayer(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          this.texture,
          0,
          layer
        );
        break;
      default:
        throw new Error();
    }

    this.isBoundToDrawFrameBuffer = true;
  }

  unbindFromDrawTexture(): void {
    if (!this.isBoundToDrawFrameBuffer) return;

    const ctx = getNNWebGLContext();
    const { gl } = ctx;

    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      null,
      0
    );

    this.isBoundToDrawFrameBuffer = false;
  }

  getDataRawFloat32(): Float32Array {
    if (
      this.textureShape.dim !== '2D' ||
      this.textureShape.type !== WebGL2RenderingContext.FLOAT
    ) {
      throw new Error();
    }
    const buf = new Float32Array(
      this.textureShape.height * this.textureShape.width * this.dimPerPixel
    );
    this.readPixels2D(buf);
    return buf;
  }

  getDataRaw():
    | { type: 'Float32Array'; buffer: Float32Array }
    | { type: 'Uint16Array'; buffer: Uint16Array }
    | { type: 'Int32Array'; buffer: Int32Array }
    | { type: 'Uint8Array'; buffer: Uint8Array } {
    // const ctx = getNNWebGLContext();
    // if (ctx.canOnlyReadRGBA && this.dimPerPixel === 1) {
    //   const packed = this.packRToRGBA();
    //   const packedData = packed.getDataRaw();
    //   packed.dispose();
    //   if (this.textureShape.internalFormat === WebGL2RenderingContext.R8UI) {
    //     // RGBA8UIが直接読めずInt32Arrayとして読まれるので、Uint8Arrayに変換してpackの有無での差異をなくす
    //     const uint8 = new Uint8Array(packedData.buffer.length);
    //     uint8.set(packedData.buffer);
    //     return { type: 'Uint8Array', buffer: uint8 };
    //   }
    //   return packedData;
    // }
    switch (this.textureShape.dim) {
      case '2D': {
        const length =
          this.textureShape.height * this.textureShape.width * this.dimPerPixel;
        switch (this.textureShape.type) {
          case WebGL2RenderingContext.FLOAT: {
            const buffer = new Float32Array(length);
            this.readPixels2D(buffer);
            return { type: 'Float32Array', buffer };
          }
          case WebGL2RenderingContext.HALF_FLOAT: {
            const buffer = new Uint16Array(length);
            this.readPixels2D(buffer);
            return { type: 'Uint16Array', buffer };
          }
          case WebGL2RenderingContext.INT: {
            const buffer = new Int32Array(length);
            this.readPixels2D(buffer);
            return { type: 'Int32Array', buffer };
          }
          case WebGL2RenderingContext.UNSIGNED_BYTE: {
            const buffer = new Uint8Array(length);
            this.readPixels2D(buffer);
            return { type: 'Uint8Array', buffer };
          }
          default:
            throw new Error();
        }
      }
      case '2DArray': {
        const sliceLength =
          this.textureShape.height * this.textureShape.width * this.dimPerPixel;
        const totalLength = sliceLength * this.textureShape.depth;
        switch (this.textureShape.type) {
          case WebGL2RenderingContext.FLOAT: {
            const buffer = new Float32Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Float32Array', buffer };
          }
          case WebGL2RenderingContext.HALF_FLOAT: {
            const buffer = new Uint16Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Uint16Array', buffer };
          }
          case WebGL2RenderingContext.INT: {
            const buffer = new Int32Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Int32Array', buffer };
          }
          case WebGL2RenderingContext.UNSIGNED_BYTE: {
            const buffer = new Uint8Array(totalLength);
            this.readPixels2DArray(
              buffer,
              sliceLength,
              this.textureShape.depth
            );
            return { type: 'Uint8Array', buffer };
          }
          default:
            throw new Error();
        }
      }
    }
  }

  setDataRaw(data: ArrayBufferView): void {
    const ctx = getNNWebGLContext();
    this.bindToReadTexture(0);
    switch (this.textureShape.dim) {
      case '2D':
        ctx.gl.texSubImage2D(
          this.target,
          0,
          0,
          0,
          this.textureShape.width,
          this.textureShape.height,
          this.textureShape.format,
          this.textureShape.type,
          data,
          0
        );
        break;
      case '2DArray':
        ctx.gl.texSubImage3D(
          this.target,
          0,
          0,
          0,
          0,
          this.textureShape.width,
          this.textureShape.height,
          this.textureShape.depth,
          this.textureShape.format,
          this.textureShape.type,
          data,
          0
        );
        break;
      default:
        throw new Error('not implemented');
    }
    this.unbindFromReadTexture();
  }

  private readPixels2D(buf: ArrayBufferView) {
    // Mac + ChromeではRチャンネルのみのテクスチャを読み出せない
    // Mac + Firefoxではさらに、RGBA8UIも読み出せない
    // packRToRGBAで基本的に回避しているが、これを経由せずRGBA8UIを直接使うコードがあるとエラーになりうる
    const ctx = getNNWebGLContext();
    this.bindToDrawTexture();
    ctx.gl.readPixels(
      0,
      0,
      this.textureShape.width,
      this.textureShape.height,
      this.textureShape.format,
      this.textureShape.type,
      buf
    );
    this.unbindFromDrawTexture();
  }

  private readPixels2DArray(
    buf: ArrayBufferView,
    sliceLength: number,
    depth: number
  ) {
    const ctx = getNNWebGLContext();
    for (let layer = 0; layer < depth; layer++) {
      this.bindToDrawTexture(layer);
      ctx.gl.readPixels(
        0,
        0,
        this.textureShape.width,
        this.textureShape.height,
        this.textureShape.format,
        this.textureShape.type,
        buf,
        sliceLength * layer
      );
      this.unbindFromDrawTexture();
    }
  }
}
// function deleteTextureWait() {
//   return new Promise<void>((resolve) => {
//     setTimeout(resolve, 1);
//   });
// }

function initWebGL() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) {
    throw new Error('WebGL2 not supported');
  }
  const allowedTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  let maxTextureSize: number;
  if (allowedTextureSize >= 16384) {
    maxTextureSize = 16384;
  } else if (allowedTextureSize >= 4096) {
    maxTextureSize = 4096;
  } else {
    throw new Error(`gl.MAX_TEXTURE_SIZE is too small (${allowedTextureSize})`);
  }
  return { gl, maxTextureSize };
}

export interface WebGLKernelInputBuffer {
  name: string;
  buffer: WebGLTensorBuffer;
}

export type WebGLKernelInput = WebGLKernelInputBuffer;

export class NNWebGLContext {
  gl: WebGL2RenderingContext;
  maxTextureSize: number;
  fb: WebGLFramebuffer;
  supportsTexture32bit: boolean;
  supportsTexture16bit: boolean;
  canReadRedTexture: boolean;
  canReadNon32bitTexture: boolean;
  private programs: Map<string, { program: WebGLProgram }> = new Map();
  private vshader!: WebGLShader;

  constructor() {
    const { gl, maxTextureSize } = initWebGL();
    this.gl = gl;
    this.maxTextureSize = maxTextureSize;

    if (gl.getExtension('EXT_color_buffer_float')) {
      // Enable color mode of gl.R32F
      this.supportsTexture32bit = true;
      // R16F is included in EXT_color_buffer_float
      // Even if this can be get, EXT_color_buffer_half_float may not be able to get in some env
      this.supportsTexture16bit = true;
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      // Enable color mode of gl.R16F
      this.supportsTexture32bit = false;
      this.supportsTexture16bit = true;
    } else {
      // Unsupported env where float texture is not supported
      throw new Error(
        'Neither EXT_color_buffer_float nor EXT_color_buffer_half_float are supported'
      );
    }
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.SAMPLE_COVERAGE);
    gl.enable(gl.SCISSOR_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    const vertexBuffer = this.createArrayBuffer(vertexArray);
    this.bindArrayBuffer(vertexBuffer);
    this.fb = nonNull(gl.createFramebuffer());
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    // バグ回避
    // Mac+Firefox, Linux+Firefoxで、RチャンネルのみのテクスチャをreadPixelsで読みだそうとするとエラーとなる
    // GL ERROR :GL_INVALID_OPERATION : glReadPixels: format and type incompatible with the current read framebuffer
    // WebGL warning: readPixels: Format and type RED/FLOAT incompatible with this R32F attachment. This framebuffer requires either RGBA/FLOAT or getParameter(IMPLEMENTATION_COLOR_READ_FORMAT/_TYPE) RGBA/FLOAT.
    const ua = navigator.userAgent;
    const problemCase =
      (ua.includes('Macintosh') && ua.includes('Firefox/')) ||
      (ua.includes('Linux') && ua.includes('Firefox/'));
    this.canReadNon32bitTexture = this.canReadRedTexture = !problemCase;
  }

  createArrayBuffer(vertexArray: Float32Array): WebGLBuffer {
    const buffer = nonNull(this.gl.createBuffer());
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, vertexArray, this.gl.STATIC_DRAW);

    return buffer;
  }

  bindArrayBuffer(buffer: WebGLBuffer): void {
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
  }

  createTexture(textureShape: TensorTextureShape): WebGLTexture {
    if (
      textureShape.dim === '2DArray' &&
      (textureShape.width === 1 || textureShape.height === 1)
    ) {
      // texSubImage3D raises Error when the following condition is met
      // WebGL: INVALID_OPERATION: texSubImage3D: ArrayBufferView not big enough for request
      // (textureShape.dim === "2DArray" && (textureShape.width === 1 || textureShape.height === 1) && textureShape.internalFormat === WebGL2RenderingContext.R16F
      throw new Error(
        'The condition raises error: textureShape.dim === "2DArray" && (textureShape.width === 1 || textureShape.height === 1))'
      );
    }
    const gl = this.gl;
    const texture = nonNull(gl.createTexture());
    gl.activeTexture(gl.TEXTURE0);
    let target: number;
    switch (textureShape.dim) {
      case '2D':
        target = gl.TEXTURE_2D;
        gl.bindTexture(target, texture);
        gl.texStorage2D(
          target,
          1,
          textureShape.internalFormat,
          textureShape.width,
          textureShape.height
        );
        break;
      case '2DArray':
        target = gl.TEXTURE_2D_ARRAY;
        gl.bindTexture(target, texture);
        gl.texStorage3D(
          target,
          1,
          textureShape.internalFormat,
          textureShape.width,
          textureShape.height,
          textureShape.depth
        );
        break;
      default:
        throw new Error();
    }
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.bindTexture(target, null);

    return texture;
  }

  createShader(type: number, source: string, name?: string): WebGLShader {
    const shader = nonNull(this.gl.createShader(type));

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);
    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      throw Error(
        `Shader Compile failed (name=${name}): ${this.gl.getShaderInfoLog(
          shader
        )}\n${source}`
      );
    }

    return shader;
  }

  addKernel(name: string, sourceCode: string): void {
    if (this.programs.has(name)) {
      return;
    }
    this.programs.set(name, { program: this.compileKernel(sourceCode, name) });
  }

  hasKernel(name: string): boolean {
    return this.programs.has(name);
  }

  compileKernel(sourceCode: string, name?: string): WebGLProgram {
    const { gl } = this;
    if (!this.vshader) {
      this.vshader = this.createShader(
        gl.VERTEX_SHADER,
        vertex_shader_source_2,
        'vertex_shader'
      );
    }
    const fshader = this.createShader(gl.FRAGMENT_SHADER, sourceCode, name),
      program = nonNull(this.gl.createProgram());

    this.gl.attachShader(program, fshader);
    this.gl.attachShader(program, this.vshader);
    this.gl.linkProgram(program);
    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      throw new Error('ShaderProgram Initialization failed.');
    }

    return program;
  }

  runKernel(
    name: string,
    inputs: WebGLKernelInput[],
    output: WebGLTensorBuffer,
    uniforms: WebGLUniformItem[],
    drawLayer: number | null = null
  ): void {
    const outputBuffer = output;
    if (outputBuffer.textureShape.dim === '2DArray' && drawLayer == null) {
      for (let d = 0; d < outputBuffer.textureShape.depth; d++) {
        this.runKernelSingleDrawLayer(name, inputs, outputBuffer, uniforms, d);
      }
    } else {
      this.runKernelSingleDrawLayer(
        name,
        inputs,
        outputBuffer,
        uniforms,
        drawLayer || 0
      );
    }
  }

  private runKernelSingleDrawLayer(
    name: string,
    inputs: WebGLKernelInput[],
    outputBuffer: WebGLTensorBuffer,
    uniforms: WebGLUniformItem[],
    drawLayer: number | null
  ): void {
    const { gl } = this;
    const kobj = this.programs.get(name);
    if (!kobj) {
      throw new Error(`Unknown kernel ${name}`);
    }

    const xyAttribLoc = gl.getAttribLocation(kobj.program, '_xy');
    for (let i = 0; i < inputs.length; i++) {
      const ini = inputs[i];
      const buffer = ini.buffer;
      buffer.bindToReadTexture(i);
    }
    outputBuffer.bindToDrawTexture(drawLayer == null ? undefined : drawLayer); // TODO

    gl.useProgram(kobj.program);

    const extendedUniforms: WebGLUniformItem[] = [...uniforms];
    if (drawLayer != null) {
      extendedUniforms.push({
        type: 'int',
        name: '_draw_depth',
        value: drawLayer,
      });
    }
    for (let i = 0; i < inputs.length; i++) {
      extendedUniforms.push({ type: 'int', name: inputs[i].name, value: i });
    }

    for (const uniform of extendedUniforms) {
      const loc = gl.getUniformLocation(kobj.program, uniform.name);
      if (loc == null) {
        continue;
      }
      switch (uniform.type) {
        case 'float':
          gl.uniform1f(loc, uniform.value);
          break;
        case 'int':
          gl.uniform1i(loc, uniform.value);
          break;
        default:
          throw new Error();
      }
    }
    gl.vertexAttribPointer(xyAttribLoc, 2, gl.FLOAT, true, 8, 0);
    gl.enableVertexAttribArray(xyAttribLoc);

    const fbStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (fbStatus !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`FRAMEBUFFER status invalid: ${fbStatus}.`);
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexArray.length / 2);

    for (let i = 0; i < inputs.length; i++) {
      const ini = inputs[i];
      const buffer = ini.buffer;
      buffer.unbindFromReadTexture();
    }

    outputBuffer.unbindFromDrawTexture();
  }
}

let context: NNWebGLContext | null = null;
export async function initializeNNWebGLContext(): Promise<void> {
  // Currently no asynchronous processing, but functional tests may be added in the future
  context = new NNWebGLContext();
}

export function getNNWebGLContext(): NNWebGLContext {
  if (!context) {
    throw new Error('WebGL Context does not exist');
  }
  return context;
}
