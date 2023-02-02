# Step 1: Import needed Python modules
import os
import numpy as np
import ctypes

# Step 2: Helper function to setup the C function interface
def setup_func(func, params, returns):

    if not isinstance(params, list):
        raise TypeError(f'params must be of type list, but got {type(params)} instead.')
    if not isinstance(returns, type(ctypes.c_int)) and returns is not None:
        raise TypeError(f'returns must be a C type or None, but got {type(returns)} instead.')

    func.argtypes = params
    func.restype  = returns

# Step 3: The initialize library function
def initialize_library(precision="single",
                       libpath=os.path.join("..", "lib", "libretina_c.so")):
    # Step 3.a: Set C types
    if precision.lower() == "single":
        c_float = ctypes.c_float
    elif precision.lower() == "double":
        c_float = ctypes.c_double
    else:
        raise ValueError(f'Unsupported precision {precision}. Supported values are "single" and "double"')
    c_int      = ctypes.c_int
    c_uint16   = ctypes.c_uint16
    c_uint16_p = ctypes.POINTER(c_uint16)
    c_float_p  = ctypes.POINTER(c_float)
    c_void_p   = ctypes.c_void_p

    # Step 3.b: Load the library
    lib = ctypes.CDLL(libpath)

    # Step 3.c: Set up all the functions in the library
    # Step 3.c.1: The Cstate_initialize function
    setup_func(lib.Cstate_initialize, [c_int, c_int, c_float, c_float, c_float], c_void_p)

    # Step 3.c.2: The Cstate_finalize function
    setup_func(lib.Cstate_finalize, [c_void_p], None)

    return lib
