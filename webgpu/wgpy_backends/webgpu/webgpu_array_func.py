from typing import List, Optional, Tuple, Union
from wgpy_backends.webgpu import common_reduction
from wgpy_backends.webgpu import common_ufunc
from wgpy_backends.webgpu.ndarray import ndarray
from wgpy_backends.webgpu.matmul import matmul_impl, tensordot_impl


class WebGPUArrayFunc:
    _instance = None

    def __init__(self) -> None:
        self.ufunc = common_ufunc
        self.reduction = common_reduction
        pass

    @staticmethod
    def instance() -> "WebGPUArrayFunc":
        if WebGPUArrayFunc._instance is None:
            WebGPUArrayFunc._instance = WebGPUArrayFunc()
        return WebGPUArrayFunc._instance

    def matmul(
        self, lhs: ndarray, rhs: ndarray, out: Optional[ndarray] = None
    ) -> ndarray:
        return matmul_impl(lhs, rhs, out)

    def tensordot(
        self,
        a: ndarray,
        b: ndarray,
        axes: Union[int, Tuple[int, int], Tuple[List[int], List[int]]] = 2,
    ) -> ndarray:
        return tensordot_impl(a, b, axes)
