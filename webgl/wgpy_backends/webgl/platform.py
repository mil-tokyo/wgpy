# platform call interface
import numpy as np
from js import gl # Pyodide-dependent

class WebGLPlatform:
    def getDeviceInfo(self) -> dict:
        return gl.getDeviceInfo().to_py()

    def createBuffer(self, buffer_id: int, texture_shape_json: str):
        return gl.createBuffer(buffer_id, texture_shape_json)

    def disposeBuffer(self, buffer_id: int):
        return gl.disposeBuffer(buffer_id)
    
    def setCommBuf(self, buffer: np.ndarray):
        return gl.setCommBuf(buffer)
    
    def setData(self, buffer_id: int, js_ctor_type: str, size: int):
        return gl.setData(buffer_id, js_ctor_type, size)

    def getData(self, buffer_id: int, js_ctor_type: str, size: int):
        return gl.getData(buffer_id, js_ctor_type, size)

    def addKernel(self, name, descriptor):
        return gl.addKernel(name, descriptor)

    def runKernel(self, descriptor):
        return gl.runKernel(descriptor)

_instance = None
def get_platform() -> WebGLPlatform:
    global _instance
    if _instance is None:
        _instance = WebGLPlatform()
    return _instance
