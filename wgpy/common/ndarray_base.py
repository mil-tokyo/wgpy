from typing import Tuple
import numpy as np

def _get_ap(x):
    return getattr(x, '__array_priority__', 0)

def _use_rhs(lhs, rhs):
    return _get_ap(lhs) < _get_ap(rhs)

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
