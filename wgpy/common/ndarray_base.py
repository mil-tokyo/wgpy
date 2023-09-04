from typing import Optional, Tuple, Union, TypeVar
import numpy as np

from wgpy.common.shape_util import args_to_tuple_of_int


def _get_ap(x):
    return getattr(x, '__array_priority__', 0)

def _use_rhs(lhs, rhs):
    return _get_ap(lhs) < _get_ap(rhs)

A = TypeVar('A', bound='NDArrayBase')

class NDArrayBase():
    __array_priority__ = 100 # dezero Variable is 200
    shape: Tuple[int, ...]
    dtype: np.dtype
    strides: Tuple[int, ...] # unit: byte
    itemsize: int
    offset: int # byte offset from buffer head
    nbytes: int

    def _binary_lhs(self, other, func, rhs):
        if _use_rhs(self, other):
            # dezero.Variableで発生
            return rhs(other)(self)
        return func(self, other)

    def __add__(self, other):
        return self._binary_lhs(other, self.array_func.ufunc.add, lambda o: o.__rmul__)

    def __sub__(self, other):
        return self._binary_lhs(other, self.array_func.ufunc.sub, lambda o: o.__rmul__)

    def __mul__(self, other):
        return self._binary_lhs(other, self.array_func.ufunc.mul, lambda o: o.__rmul__)

    def __truediv__(self, other):
        return self._binary_lhs(other, self.array_func.ufunc.truediv, lambda o: o.__rmul__)

    def __pow__(self, other):
        return self._binary_lhs(other, self.array_func.ufunc.pow, lambda o: o.__rmul__)

    def __lt__(self, other):
        return self.array_func.ufunc.lt(self, other)

    def __le__(self, other):
        return self.array_func.ufunc.le(self, other)

    def __eq__(self, other):
        return self.array_func.ufunc.eq(self, other)

    def __ne__(self, other):
        return self.array_func.ufunc.ne(self, other)

    def __ge__(self, other):
        return self.array_func.ufunc.ge(self, other)

    def __gt__(self, other):
        return self.array_func.ufunc.gt(self, other)

    def __matmul__(self, other):
        return self.array_func.matmul(self, other)
    
    # TODO: floordiv, mod, divmod, lshift, rshift, and, xor, or

    def __radd__(self, other):
        return self.array_func.ufunc.add(other, self)

    def __rsub__(self, other):
        return self.array_func.ufunc.sub(other, self)

    def __rmul__(self, other):
        return self.array_func.ufunc.mul(other, self)

    def __rtruediv__(self, other):
        return self.array_func.ufunc.truediv(other, self)

    def __rpow__(self, other):
        return self.array_func.ufunc.pow(other, self)

    def __rmatmul__(self, other):
        return self.array_func.matmul(other, self)

    # TODO: rfloordiv, rmod, rdivmod, rlshift, rrshift, rand, rxor, ror

    def __iadd__(self, other):
        return self.array_func.ufunc.add(self, other, out=self)

    def __isub__(self, other):
        return self.array_func.ufunc.sub(self, other, out=self)

    def __imul__(self, other):
        return self.array_func.ufunc.mul(self, other, out=self)

    def __itruediv__(self, other):
        return self.array_func.ufunc.truediv(self, other, out=self)

    def __ipow__(self, other):
        return self.array_func.ufunc.pow(self, other, out=self)

    # TODO: ifloordiv, imod, ilshift, irshift, iand, ixor, ior

    def __pos__(self):
        return self.array_func.ufunc.pos(self)

    def __neg__(self):
        return self.array_func.ufunc.neg(self)
    
    def __abs__(self):
        return self.array_func.ufunc.abs(self)

    def __invert__(self):
        return self.array_func.ufunc.invert(self)

    def __float__(self):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return float(self.get())
    
    def __int__(self):
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return int(self.get())

    def __len__(self):
        if self.ndim > 0:
            return self.shape[0]
        else:
            raise TypeError('len() of unsized object')

    def _check_c_contiguous(self) -> bool:
        stride = self.itemsize
        for d in range(self.ndim-1, -1, -1):
            if stride != self.strides[d]:
                return False
            stride *= self.shape[d]
        return True
    
    def _check_f_contiguous(self) -> bool:
        stride = self.itemsize
        for d in range(self.ndim):
            if stride != self.strides[d]:
                return False
            stride *= self.shape[d]
        return True

    @property
    def T(self):
        return self.transpose()

    def get(self, stream=None, order='C', out=None) -> np.ndarray:
        # ignore stream
        assert order == 'C'
        assert out is None
        return self.get_data()

    def to_cpu(self) -> np.ndarray:
        return self.get()
    
    def _is_full_view(self) -> bool:
        if self.offset != 0:
            return False
        if self.size != self.buffer.size:
            return False
        return True
    
    def get_view(self: A, shape: Tuple[int, ...], dtype: np.dtype, strides: Tuple[int, ...], offset: int) -> A:
        raise NotImplementedError

    def broadcast_to(self: A, shape: Union[tuple, int], subok=False) -> A:
        if isinstance(shape, int):
            shape = (shape,)
        orig_shape = list(self.shape)
        assert len(orig_shape) <= len(shape)
        new_strides = list(self.strides)
        while len(orig_shape) < len(shape):
            # append 1 to head
            orig_shape.insert(0, 1)
            new_strides.insert(0, 0)
        for d in range(len(shape)):
            if orig_shape[d] == 1 and shape[d] != 1:
                # broadcasted axis
                new_strides[d] = 0
        return self.get_view(shape, self.dtype, tuple(new_strides), self.offset)

    def dot(self, other, out=None):
        # TODO: avoid circular reference
        from wgpy.binary import dot
        return dot(self, other, out=out)

    def max(self, *args, **kwargs):
        from wgpy.reduction import max
        return max(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        from wgpy.construct import asarray, asnumpy
        return asarray(asnumpy(self).argmax(*args, **kwargs))

    def reshape(self, *shape):
        # array.reshape(2,3) and array.reshape((2,3)) are ok
        from wgpy.manipulation import reshape
        return reshape(self, args_to_tuple_of_int(shape))

    def sum(self, *args, **kwargs):
        from wgpy.reduction import sum
        return sum(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        from wgpy.reduction import mean
        return mean(self, *args, **kwargs)

    def var(self, *args, **kwargs):
        from wgpy.reduction import var
        return var(self, *args, **kwargs)

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]]=None):
        from wgpy.manipulation import squeeze
        return squeeze(self, axis)

    def transpose(self, *axes):
        from wgpy.manipulation import transpose
        return transpose(self, args_to_tuple_of_int(axes))

    def ravel(self):
        from wgpy.manipulation import ravel
        return ravel(self)
    
    def reduced_view(self: A) -> A:
        # not exist in numpy, but exist in cupy
        # https://github.com/cupy/cupy/issues/2114
        if self.ndim == 0:
            # shape == () => ()
            shape = ()
            strides = ()
        elif self.ndim == 1:
            shape = self.shape
            strides = self.strides
        elif self.size == 0:
            shape = (0,)
            strides = (self.itemsize,)
        elif self.size == 1:
            # when ndim==1, shape=(1,)
            shape = ()
            strides = ()
        else:
            # generic case
            new_shape = list(self.shape)
            new_strides = list(self.strides)
            while True:
                changed = False
                for d in range(len(new_shape)-1):
                    if new_strides[d] == new_strides[d+1] * new_shape[d+1]:
                        # dim d and d+1 can be combined
                        new_shape[d] = new_shape[d] * new_shape[d+1]
                        new_shape.pop(d+1)
                        new_strides.pop(d)
                        changed = True
                        break
                if not changed:
                    break
            shape = tuple(new_shape)
            strides = tuple(new_strides)
        return self.get_view(shape, dtype=self.dtype, strides=strides, offset=self.offset)
