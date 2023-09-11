# platform call interface
import numpy as np
from js import gpu # Pyodide-dependent

class WebGPUPlatform:
    def __init__(self) -> None:
        self._latest_comm_buf = None

    def getDeviceInfo(self) -> dict:
        return gpu.getDeviceInfo().to_py()

    def createBuffer(self, buffer_id: int, byte_length: int):
        return gpu.createBuffer(buffer_id, byte_length)

    def createMetaBuffer(self, buffer_id: int, byte_length: int):
        return gpu.createMetaBuffer(buffer_id, byte_length)

    def disposeBuffer(self, buffer_id: int):
        return gpu.disposeBuffer(buffer_id)
    
    def setCommBuf(self, buffer: np.ndarray):
        self._latest_comm_buf = buffer
        return gpu.setCommBuf(buffer)
    
    def setData(self, buffer_id: int, byte_length: int):
        if not gpu.setData(buffer_id, byte_length):
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            if not gpu.setData(buffer_id, byte_length):
                raise ValueError("setData failed twice")

    def getData(self, buffer_id: int, byte_length: int):
        if not gpu.getData(buffer_id, byte_length):
            self.setCommBuf(self._latest_comm_buf)
            if not gpu.getData(buffer_id, byte_length):
                raise ValueError("getData failed twice")

    def addKernel(self, name, descriptor):
        return gpu.addKernel(name, descriptor)

    def runKernel(self, descriptor):
        return gpu.runKernel(descriptor)

_instance = None
def get_platform() -> WebGPUPlatform:
    global _instance
    if _instance is None:
        _instance = WebGPUPlatform()
    return _instance
