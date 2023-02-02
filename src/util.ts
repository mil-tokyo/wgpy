export function arrayEqual(
  a: readonly unknown[],
  b: readonly unknown[]
): boolean {
  if (!a || !b) {
    return false;
  }
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

export function arraySum(vec: ArrayLike<number>): number {
  let x = 0;
  for (let i = 0; i < vec.length; i++) {
    x += vec[i];
  }
  return x;
}

export function arrayProd(vec: ArrayLike<number>): number {
  let x = 1;
  for (let i = 0; i < vec.length; i++) {
    x *= vec[i];
  }
  return x;
}

export function nonNull<T>(v: T | null | undefined): T {
  if (!v) {
    throw new Error();
  }
  return v;
}

export function arange(stop: number): number[];
export function arange(start: number, stop: number): number[];
export function arange(start: number, stop: number, step: number): number[];
export function arange(start: number, stop?: number, step = 1): number[] {
  if (stop == null) {
    const len = start;
    const array = new Array(len);
    for (let i = 0; i < len; i++) {
      array[i] = i;
    }
    return array;
  } else {
    const array: number[] = [];
    if (step > 0) {
      for (let i = start; i < stop; i += step) {
        array.push(i);
      }
    } else {
      for (let i = start; i > stop; i += step) {
        array.push(i);
      }
    }
    return array;
  }
}

export function base64ToUint8Array(encodedData: string): Uint8Array {
  const decoded = window.atob(encodedData);
  const array = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) {
    array[i] = decoded.charCodeAt(i);
  }
  return array;
}

export function uint8ArrayToBase64(array: Uint8Array): string {
  const binaryStr = String.fromCharCode.apply(null, Array.from(array));
  return window.btoa(binaryStr);
}
