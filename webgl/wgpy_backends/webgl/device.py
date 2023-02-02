class Device:
    def __init__(self, device=None) -> None:
        self.id = 0 # single device

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __int__(self):
        return self.id

    def get_device_id(self):
        return self.id

device = Device()
