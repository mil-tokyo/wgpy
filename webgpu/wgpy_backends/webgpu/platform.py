# platform call interface
import numpy as np
from js import gpu # Pyodide-dependent

class WebGPUPlatform:
    def getDeviceInfo(self) -> dict:
        return gpu.getDeviceInfo().to_py()

    def createBuffer(self, buffer_id: int, byte_length: int):
        return gpu.createBuffer(buffer_id, byte_length)

    def disposeBuffer(self, buffer_id: int):
        return gpu.disposeBuffer(buffer_id)
    
    def setCommBuf(self, buffer: np.ndarray):
        return gpu.setCommBuf(buffer)
    
    def setData(self, buffer_id: int, byte_length: int):
        return gpu.setData(buffer_id, byte_length)

    def getData(self, buffer_id: int, byte_length: int):
        return gpu.getData(buffer_id, byte_length)

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
