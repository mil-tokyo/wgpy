import numpy as np
from wgpy.construct import asarray
from wgpy.manipulation import broadcast_to


class broadcast:
    def __init__(self, *arrays):
        arrays = [asarray(array) for array in arrays]  # element may be scalar (float)
        self.shape = np.broadcast_shapes(*[array.shape for array in arrays])
        self.nd = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.values = [broadcast_to(array, self.shape) for array in arrays]


__all__ = ["broadcast"]
