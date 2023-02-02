/**
 * Data type of tensor element.
 */
export type DType = 'float32' | 'int32' | 'uint8' | 'bool';
export const DTypeDefault: DType = 'float32';
export type TypedArrayTypes = Float32Array | Int32Array | Uint8Array;
export type TypedArrayConstructor =
  | Float32ArrayConstructor
  | Int32ArrayConstructor
  | Uint8ArrayConstructor;
export const TypedArrayForDType: {
  [dtype in DType]: TypedArrayConstructor;
} = {
  float32: Float32Array,
  int32: Int32Array,
  uint8: Uint8Array,
  bool: Uint8Array,
};
