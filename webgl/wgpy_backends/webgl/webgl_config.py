from wgpy_backends.webgl.platform import get_platform

_info_loaded = False
_float_texture_bit = 32
_max_texture_size = 1
_can_read_r_texture = False
_can_read_non_32bit_texture = False
def _get_info():
    global _info_loaded, _float_texture_bit, _max_texture_size, _can_read_r_texture, _can_read_non_32bit_texture
    if _info_loaded:
        return
    info = get_platform().getDeviceInfo()
    if info['supportsTexture32bit']:
        _float_texture_bit = 32
    elif info['supportsTexture16bit']:
        _float_texture_bit = 16
    else:
        # This case, constructor of NNWebGLContext fails
        raise ValueError()
    _max_texture_size = info['maxTextureSize']
    _can_read_r_texture = info['canReadRedTexture']
    _can_read_non_32bit_texture = info['canReadNon32bitTexture']
    _info_loaded = True

def get_float_texture_bit():
    _get_info()
    return _float_texture_bit

def get_max_texture_size() -> int:
    _get_info()
    return _max_texture_size

def can_read_r_texture():
    # If the environment can call readPixels of texture of R channel only.
    _get_info()
    return _can_read_r_texture

def can_read_non_32bit_texture():
    # If the environment can call readPixels of texture of non-32bit element (HALF_FLOAT, UNSIGNED_BYTE).
    _get_info()
    return _can_read_non_32bit_texture
