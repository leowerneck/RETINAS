# Step 1: Import needed Python modules
from ctypes import c_int, c_uint16, c_float, c_double
from ctypes import c_void_p, CDLL, POINTER
from numpy import single, double

# Step 2: Helper function to setup the C function interface
def setup_func(func, params, returns):
    """
      Setup the C functions

      Inputs
      ------
        func : ctypes._FuncPtr
          Pointer to a C function.

        params : list of _ctypes.PyCSimpleType
          List of C types for function parameters.

        returns : _ctypes.PyCSimpleType or None
          Return value of the C function.

      Returns
      -------
        Nothing.

      Raises
      ------
        TypeError : If params have the wrong type.
        TypeError : If any param in params have the wrong type.
        TypeError : If returns have the wrong type.
    """

    # Step 2.a: Check inputs have the correct types
    Ctype    = type(c_int)
    Cptrtype = type(POINTER(c_int))
    if not isinstance(params, list):
        raise TypeError(f'params must be of type list, but got {type(params)} instead.')
    for param in params:
        if not isinstance(param, Ctype) and not isinstance(param, Cptrtype):
            raise TypeError(f'params must be a list of C types, but found {type(param)}.')
    if not isinstance(returns, Ctype) and returns is not None:
        raise TypeError(f'returns must be a C type or None, but got {type(returns)} instead.')

    # Step 2.b: Setup the types of function parameters and return value
    func.argtypes = params
    func.restype  = returns

# Step 3: The initialize library function
def initialize_library(libpath, real=float, c_real=c_float):
    """
      Initialize the library.

      Inputs
      ------
        libpath : str
          Path to the library.

        real : type
          Type of float to use (default is float).

        c_real : ctype
          Type of C float to use (default is c_float).

      Returns
      -------
        lib : ctypes.CDLL
          Library interface file, fully initialized.

      Raises
      ------
        ValueError : If precision has an unsupported value.
    """
    # Step 3.a: Set C types
    c_uint16_p = POINTER(c_uint16)
    c_real_p   = POINTER(c_real)

    # Step 3.b: Load the library
    lib = CDLL(libpath)

    # Step 3.c: Set up all the functions in the library
    # Step 3.c.1: The state_initialize function
    # state_struct *state_initialize(
    #     const int N_horizontal,
    #     const int N_vertical,
    #     const REAL upsample_factor,
    #     const REAL A0,
    #     const REAL B1 );
    setup_func(lib.state_initialize, [c_int, c_int, c_real, c_real, c_real], c_void_p)

    # Step 3.c.2: The state_finalize function
    # void state_finalize( state_struct *restrict state );
    setup_func(lib.state_finalize, [c_void_p], None)

    # Step 3.c.3: The typecast_input_image_and_compute_brightness function
    # REAL typecast_input_image_and_compute_brightness(
    #     const uint16_t *restrict input_array,
    #     state_struct *restrict state );
    setup_func(lib.typecast_input_image_and_compute_brightness,
               [c_uint16_p, c_void_p], c_real)

    # Step 3.c.4: The typecast_input_image_rebin_4x4_and_compute_brightness function
    # REAL typecast_input_image_rebin_4x4_and_compute_brightness(
    #     const uint16_t *restrict input_array,
    #     state_struct *restrict state );
    setup_func(lib.typecast_input_image_and_compute_brightness,
               [c_uint16_p, c_void_p], c_real)

    # Step 3.c.5: The set_zeroth_eigenframe function
    # void set_zeroth_eigenframe( state_struct *restrict state );
    setup_func(lib.set_zeroth_eigenframe, [c_void_p], None)

    # Step 3.c.6: The cross_correlate_and_compute_displacements function
    # void cross_correlate_and_compute_displacements(
    #     state_struct *restrict state,
    #     REAL *restrict displacements );
    setup_func(lib.cross_correlate_and_compute_displacements,
               [c_void_p, c_real_p], None)

    # Step 3.c.7: The upsample_and_compute_subpixel_displacements function
    # void upsample_and_compute_subpixel_displacements(
    #     state_struct *restrict state,
    #     REAL *restrict displacements );
    setup_func(lib.upsample_and_compute_subpixel_displacements,
               [c_void_p, c_real_p], None)

    # Step 3.c.8: The build_next_eigenframe function
    # void build_next_eigenframe(
    #     const REAL *restrict displacements,
    #     state_struct *restrict state );
    setup_func(lib.build_next_eigenframe, [c_real_p, c_void_p], None)

    # Step 3.c.8: The compute_displacements_and_build_next_eigenframe function
    # void compute_displacements_and_build_next_eigenframe(
    #     state_struct *restrict state,
    #     REAL *restrict displacements );
    setup_func(lib.compute_displacements_and_build_next_eigenframe,
               [c_void_p, c_real_p], None)

    return lib
