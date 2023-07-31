import numpy as np

# For future use such as version specifier
header = """
"""

native_scalar_type_for_dtype = {
    np.dtype(np.float32): 'f32',
    np.dtype(np.int32): 'i32',
    np.dtype(np.uint8): 'u32',
    np.dtype(np.bool_): 'bool',
}

native_scalar_type_to_default_dtype = {
    'f32': np.dtype(np.float32),
    'i32': np.dtype(np.int32),
    'u32': np.dtype(np.uint8),
    'bool': np.dtype(np.bool_),
}

