from wgpy_backends.runtime.device import Device, device


class Event:
    pass


class Stream:
    pass


class MemoryPool:
    pass


class PinnedMemoryPool:
    pass


# cuda.cupy.cuda.get_device_id()
def get_device_id():
    return device.get_device_id()
