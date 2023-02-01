#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Real-time image cross-correlation library
# 2022-06-22
# Adapted on 2022-07-07 by Ector Ayala, Megan Nolan, Austin Brandenberger, and Leo Werneck
# Step 1: Import needed Python modules
import os, ctypes
import numpy as np

# Step 2: Set useful C data types
# Step 2.a: Basic data types
c_int    = ctypes.c_int
c_uint16 = ctypes.c_uint16
c_float  = ctypes.c_double

# Step 2.b: Pointers
c_void_p   = ctypes.c_void_p
c_uint16_p = ctypes.POINTER(c_uint16)
c_float_p  = ctypes.POINTER(c_float)

# Step 3: Load the image analysis C library
lib = ctypes.CDLL(os.path.join(".","cross_correlation_eigenshot_CUDA","libimageanalysis.so"))

# Step 4: Set up all the functions in the library
# Step 4.a: cross_correlate_and_build_next_eigenframe
# int cross_correlate_and_build_next_eigenframe( CUDAstate_struct *restrict CUDAstate,
#                                                REAL *restrict displacement );
lib.cross_correlate_and_build_next_eigenframe.argtypes = [c_void_p, c_float_p]
lib.cross_correlate_and_build_next_eigenframe.restype  = c_int

# Step 4.b: initialize_CUDAstate
# CUDAstate_struct *initialize_CUDAstate( const int N_horizontal,
#                                   const int N_vertical,
#                                   const REAL upsample_factor,
#                                   const REAL A0,
#                                   const REAL B1 );
lib.initialize_CUDAstate.argtypes = [c_int, c_int, c_float, c_float, c_float, c_float]
lib.initialize_CUDAstate.restype  = c_void_p

# Step 4.c: finalize_CUDAstate
# int finalize_CUDAstate( CUDAstate_struct *restrict CUDAstate );
lib.finalize_CUDAstate.argtypes = [c_void_p]
lib.finalize_CUDAstate.restype  = c_int

# Step 4.d: typecast_and_return_brightness
# REAL typecast_and_return_brightness( const uint16_t *restrict input_array,
#                                      CUDAstate_struct *restrict CUDAstate );
lib.typecast_and_return_brightness.argtypes = [c_uint16_p, c_void_p]
lib.typecast_and_return_brightness.restype  = c_float

# Step 4.e: typecast_sum_and_rebin_image
# REAL typecast_rebin_4x4_and_return_brightness( const uint16_t *restrict input_array,
#                                                CUDAstate_struct *restrict CUDAstate );
lib.typecast_rebin_4x4_and_return_brightness.argtypes = [c_uint16_p, c_void_p]
lib.typecast_rebin_4x4_and_return_brightness.restype  = c_float

# Step 4.f: set_zeroth_eigenframe
# int set_zeroth_eigenframe( CUDAstate_struct *restrict CUDAstate );
lib.set_zeroth_eigenframe.argtypes = [c_void_p]
lib.set_zeroth_eigenframe.restype  = c_int

lib.dump_eigenframe.argtypes = [c_void_p]
lib.dump_eigenframe.restype  = c_int

def dump_eigenframe(state):
    err = lib.dump_eigenframe(state.CUDAstate)

# Step 5: Define the CrossCorrelationState class
class CrossCorrelationState(object):
    pass


def center_array_max(a):
    # fast recentering of an image array around the maximum pixel
    shape = a.shape
    ind_max = np.unravel_index(np.argmax(a, axis=None), shape)
    move_0 = int(np.rint(shape[0]/2 - (ind_max[0]+0.5)+0.1))
    move_1 = int(np.rint(shape[1]/2 - (ind_max[1]+0.5)+0.1))
    return np.roll(a, (move_0, move_1), axis=(0, 1)), float(-move_0), float(-move_1)


# Step 6: This function pre-processes the image, performing
#         a typecast from uint16 to float and returning the
#         brightness of the image
def preprocess_cross_correlation_data(state,image_array):
    if state.is_first_frame:
        image_array, state.v_0, state.h_0 = center_array_max(image_array)

    # Step 6.a: Get pointer to image_array
    image_array_p = image_array.ctypes.data

    # Step 6.b: Now call the C function typecast_and_return_brightness()
    brightness = lib.typecast_and_return_brightness(ctypes.cast(image_array_p,c_uint16_p),
                                                    state.CUDAstate)

    # Step 6.c: The image is now in State.CUDAstate.new_image_time_domain,
    #           which can only be accessed in the C code.
    return brightness

# Step 7: This function pre-processes the image, performing
#         a typecast from uint16 to float, rebinning it by
#         grouping 4x4 pixel blocks, and returning the
#         brightness of the image
def preprocess_cross_correlation_data_4x4_rebin(state,image_array):
    if state.is_first_frame:
        image_array, state.v_0, state.h_0 = center_array_max(image_array)

    # Step 7.a: Get pointer to image_array
    image_array_p = image_array.ctypes.data

    # Step 7.b: Now call the C function typecast_rebin_4x4_and_return_brightness()
    brightness = lib.typecast_rebin_4x4_and_return_brightness(ctypes.cast(image_array_p,c_uint16_p),
                                                              state.CUDAstate)

    # Step 7.c: The image is now in State.CUDAstate.new_image_time_domain,
    #           which can only be accessed in the C code.
    return brightness

# Step 8: This function initializes the state object
def initialize_cross_correlation(N_horizontal, N_vertical, time_constant, upsample_factor, offset, **kwargs):
    """
    .------------------------------.
    | initialize_cross_correlation |
    .------------------------------.

    N_horizontal     = n pixels in horizontal directon
    N_vertical       = n  pixels in vertical directon
    time_constant    = time constant for IIR filter, used to calculate decay constant (x) and recursion coeffs (A0, B1)
    upsample_factor  = Amount to upsample the given pixel resolution

    """
    # Step 8.a: Initialize the state object
    state = CrossCorrelationState()

    # Step 8.b: Compute x based on the time constant
    x = np.exp(-1/time_constant)

    # Step 8.c: Set state parameters that are used in Python
    state.A0              = 1-x
    state.B1              = x
    state.is_first_frame  = True
    state.N_horizontal    = N_horizontal
    state.N_vertical      = N_vertical
    state.upsample_factor = upsample_factor
    state.offset          = offset

    # Step 8.d: Set the C state using the initialize_CUDAstate() function
    state.CUDAstate = lib.initialize_CUDAstate(N_horizontal, N_vertical, upsample_factor, state.A0, state.B1, offset)

    # Step 8.e: Return the state object
    return state


# Step 9: This functions performs the cross correlation
#         and updates the eigenframe.
def analyze_cross_correlation(state):
    # cross-correlate the eigenframe and new_image and update the eigenframe in state
    # return the displacement of the particle in new_image
    """
    .---------------------------.
    | analyze_cross_correlation |
    .---------------------------.
    Function inputs:
        -state     = object of the CrossCorrelationsState class. Contains all the necessary
                    parameters our code requires
    Function output:
        -displacement values of the current frame compared to the previous eigenframe
    """
    # Step 9.a: Initialize the displacements to initial offset
    displacements = np.zeros(2,dtype=np.float64)

    # Step 9.b: Perform the analysis
    if state.is_first_frame:
        # Step 9.b.i: If in first frame, compute the FFT of the
        #           image and store it as the zeroth eigenframe
        key_err = lib.set_zeroth_eigenframe(state.CUDAstate)
        state.is_first_frame = False
        displacements = np.array([state.h_0, state.v_0])
    else:
        # Step 9.b.ii: For all other frames, compute the displacement using
        #           the cross correlation function with image upsampling
        displacements_p = displacements.ctypes.data
        key_err         = lib.cross_correlate_and_build_next_eigenframe(state.CUDAstate,
                                                                        ctypes.cast(displacements_p,c_float_p))
    # Step 9.c: Return the displacements
    return displacements

# Step 10: Finalize the cross correlation by destroying
#          the CUDAstate object, freeing memory in the process
def terminate_cross_correlation(state):
    key_err = lib.finalize_CUDAstate(state.CUDAstate)
    if key_err != 0:
        print("ERROR in finalize_cross_correlation")
