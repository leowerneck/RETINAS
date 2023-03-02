#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Real-time image cross-correlation library
# 2022-06-22
# Adapted on 2022-06-27 by Ector Ayala, Megan Nolan, and Austin Brandenberger
# Step 0: Import needed Python modules
import os #,shutil,time

import numpy as np

# Step 0.a: Check if NVIDIA GPU is available
#           Source: https://github.com/BVLC/caffe/issues/6825
"""
  Check NVIDIA with nvidia-smi command
  Returning code 0 if no error, it means NVIDIA is installed
  Other codes mean not installed
"""
global nvidia_gpu_detected
code = os.system('nvidia-smi >/dev/null 2>&1')
if code == 0 :
    nvidia_gpu_detected = True
else:
    nvidia_gpu_detected = False

analysis_mode = 'cpu'
if analysis_mode == 'cpu':
    import numpy as np_or_cp
else:
    import cupy as np_or_cp


# Step 1: Defining auxiliary functions-------------------------
# phase_cross_correlation function, adapted for use with CPUs or GPUs
def phase_cross_correlation(ref_freq, new_frame_freq, upsample_factor):
    """
    .-------------------------.
    | phase_cross_correlation |
    .-------------------------.

    This is an adapted version of the original, scikit-image function. For full
    documentation, please look at the original function by going to:
    https://scikit-image.org/docs/dev/api/skimage.registration.html?highlight=phase_cross_correlation#skimage.registration.phase_cross_correlation

    The original source code can be found at:
    https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py
    """
    # Step 2.ii: Store the shape of the images
    shape = new_frame_freq.shape

    # Step 2.iii: Compute the cross-correlation between
    #             the reference and moving images
    image_prod = new_frame_freq * ref_freq.conj()
    cc_image   = np_or_cp.fft.ifft2(image_prod)

    minima    = np_or_cp.unravel_index(np_or_cp.argmin(np_or_cp.abs(cc_image)), cc_image.shape) # MINIMUM!!!
    midpoints = np_or_cp.array([np_or_cp.fix(axis_size / 2) for axis_size in shape])

    # rfmin = ref_freq[      minima[0]][minima[1]]
    # nfmin = new_frame_freq[minima[0]][minima[1]]
    # ipmin = image_prod[    minima[0]][minima[1]]
    # ccmin = cc_image[      minima[0]][minima[1]]
    # print(f"Minima before upsampling: {minima[1]:.15e} {minima[0]:.15e}")
    # print(f"ref_freq   at minima    : {rfmin.real:.15e} {rfmin.imag:.15e}")
    # print(f"new_frame  at minima    : {nfmin.real:.15e} {nfmin.imag:.15e}")
    # print(f"image_prod at minima    : {ipmin.real:.15e} {ipmin.imag:.15e}")
    # print(f"cc_image   at minima    : {ccmin.real:.15e} {ccmin.imag:.15e}")

    shifts = np_or_cp.array(minima, dtype=np_or_cp.float32)

    shifts[shifts > midpoints] -= np_or_cp.array(shape)[shifts > midpoints]

    shifts                = np_or_cp.around(shifts * upsample_factor) / upsample_factor
    upsampled_region_size = np_or_cp.ceil(upsample_factor * 1.5)

    dftshift        = np_or_cp.fix(upsampled_region_size / 2.0)
    upsample_factor = np_or_cp.array(upsample_factor, dtype=np_or_cp.float32)

    sample_region_offset  = dftshift - shifts * upsample_factor

    data                  = image_prod.conj()
    upsampled_region_size = [upsampled_region_size, ] * data.ndim
    im2pi                 = complex(0, 2 * np_or_cp.pi)
    dim_properties        = list(zip(data.shape, upsampled_region_size, sample_region_offset))
    #image_product = data
    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = (np_or_cp.arange(ups_size) - ax_offset)[:, None] * np_or_cp.fft.fftfreq(n_items, upsample_factor)
        kernel = np_or_cp.exp(-im2pi * kernel)
        data = np_or_cp.tensordot(kernel, data, axes=(1, -1))

    cc_image_new = data.conj()
    minima = np_or_cp.unravel_index(np_or_cp.argmin(np_or_cp.abs(cc_image_new)), cc_image_new.shape)

    for dim in range(new_frame_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    minima = np_or_cp.stack(minima).astype(np_or_cp.float32) - dftshift
    shifts = shifts + minima / upsample_factor
    # print(f"Displacement after upsampling : {minima[0]:.15e} {minima[1]:.15e}")

    return shifts


# The freq_shift function returns the "new" frame shifted to the Eigenframe in the Fourier Domain

def freq_shift(new_frame_freq, horizontal_translation, vertical_translation, N_horizontal, N_vertical):
    """
    .------------.
    | freq_shift |
    .------------.

    This function shifts a given frame onto our eigenframe in the Fourier domain.

    Function inputs:
        - new_frame_freq    : Current frame data in Fourier Domain
        - nz                : Number of pixels in z-direction
        - ny                : Number of pixels in y-direction
    Function output:
        - the current (present) frame shifted onto our currentz eigenframe
    """
    #Initialize shift arrays.
    shift_horizontal = []
    shift_vertical = []

    # No negative in exp() since we're shifting *back* to the first frame.
    #             The if condition is used to match the weird array format in the Matlab code.
    #             Note also that Matlab starts index arrays at 1 rather than 0.
    for i_vertical in range(N_vertical):
        if i_vertical < N_vertical/2:
            shift_vertical.append(np_or_cp.exp(2.*np_or_cp.pi*complex(0,1)*vertical_translation*(i_vertical/N_vertical)))
        else:
            shift_vertical.append(np_or_cp.exp(2.*np_or_cp.pi*complex(0,1)*vertical_translation*((i_vertical-N_vertical)/N_vertical)))
    shift_vertical = np_or_cp.array(shift_vertical)
    conjugate_index = int(N_vertical/2)
    shift_vertical[conjugate_index] = np_or_cp.real(shift_vertical[conjugate_index])

    for i_horizontal in range(N_horizontal):
        if i_horizontal < N_horizontal/2:
            shift_horizontal.append(np_or_cp.exp(2.*np_or_cp.pi*complex(0,1)*horizontal_translation*(i_horizontal/N_horizontal)))
        else:
            shift_horizontal.append(np_or_cp.exp(2.*np_or_cp.pi*complex(0,1)*horizontal_translation*((i_horizontal-N_horizontal)/N_horizontal)))
    shift_horizontal = np_or_cp.array(shift_horizontal)
    conjugate_index = int(N_horizontal/2)
    shift_horizontal[conjugate_index] = np_or_cp.real(shift_horizontal[conjugate_index])

    #Again, on small examples the Matlab code is actually doing an outer product.
    shift_all = np_or_cp.outer(shift_vertical,shift_horizontal)

    #Create the phase-shifted frame.
    shifted_frame = np_or_cp.multiply(new_frame_freq,shift_all)
    return shifted_frame


class CrossCorrelationState(object):
    pass


def rebin(arr, binning, dtype=np_or_cp.uint16):
    shape = (arr.shape[0]//binning, binning,
             arr.shape[1]//binning, binning)
    return arr.reshape(shape).sum(axis=(-1,1), dtype=dtype)


def center_array_max(a):
    # fast recentering of an image array around the maximum pixel
    shape = a.shape
    ind_max = np_or_cp.unravel_index(np_or_cp.argmax(a, axis=None), shape)
    move_0 = int(np_or_cp.rint(shape[0]/2 - (ind_max[0]+0.5)+0.1))
    move_1 = int(np_or_cp.rint(shape[1]/2 - (ind_max[1]+0.5)+0.1))
    return np_or_cp.roll(a, (move_0, move_1), axis=(0, 1)), float(-move_0), float(-move_1)


def center_array_min(a):
    # fast recentering of an image array around the maximum pixel
    shape = a.shape
    ind_min = np_or_cp.unravel_index(np_or_cp.argmin(a, axis=None), shape)
    move_0 = int(np_or_cp.rint(shape[0]/2 - (ind_min[0]+0.5)+0.1))
    move_1 = int(np_or_cp.rint(shape[1]/2 - (ind_min[1]+0.5)+0.1))
    return np_or_cp.roll(a, (move_0, move_1), axis=(0, 1)), float(-move_0), float(-move_1)


# Step 3: Setup ---------------------------------------------------------------#

def initialize_cross_correlation(N_horizontal, N_vertical, time_constant, upsample_factor, offset, center=None, **kwargs):
    """
    .------------------------------.
    | initialize_cross_correlation |
    .------------------------------.

    N_horizontal     = n pixels in horizontal directon
    N_vertical       = n  pixels in vertical directon
    time_constant    = time constant for IIR filter, used to calculate decay constant (x) and recursion coeffs (A0, B1)
    upsample_factor  = Amount to upsample the given pixel resolution
    center           = "min", "max", or None. How to recenter the first image
    """
    # setup parameters and allocate memory
    state = CrossCorrelationState()

    x = np_or_cp.exp(-1/time_constant)
    state.A0 = 1-x
    state.B1 = x
    #I'm not sure how to use this definition of the eigenframe, it's just zeros?
    state.eigenframe = np_or_cp.zeros(shape=(N_vertical, N_horizontal))
    state.is_first_frame = True

    #save N_vertical, N_horizontal to use in freq_shift function. Upsample factor to use in phase_cross_correlation. Offset to use in shot noise weighting.
    state.N_horizontal = N_horizontal
    state.N_vertical = N_vertical
    state.upsample_factor = upsample_factor
    state.offset = offset
    state.center = center

    return state


def preprocess_cross_correlation_data(state, image_array):
    if state.is_first_frame:
        if state.center == "max":
            image_array, state.v_0, state.h_0 = center_array_max(image_array)
        elif state.center == "min":
            image_array, state.v_0, state.h_0 = center_array_min(image_array)
        else:
            state.h_0 = 0.0 
            state.v_0 = 0.0
    # calculate max brightness and convert to float
    brightness = np_or_cp.sum(image_array, dtype=np_or_cp.uint64)
    state.preprocessed_image_array = image_array.astype(np_or_cp.float32)
    return brightness


def preprocess_cross_correlation_data_4x4_rebin(state, image_array):
    if state.is_first_frame:
        if state.center == "max":
            image_array, state.v_0, state.h_0 = center_array_max(image_array)
        elif state.center == "min":
            image_array, state.v_0, state.h_0 = center_array_min(image_array)
        else:
            state.h_0 = 0.0 
            state.v_0 = 0.0
    # calculate max brightness, bin 4x4, and convert to float
    brightness = np_or_cp.sum(image_array, dtype=np_or_cp.uint64)
    preprocessed_image_array = rebin(image_array, 4)
    state.preprocessed_image_array = preprocessed_image_array.astype(np_or_cp.float32)
    return brightness


#Step 4: The Analysis----------------------------------------------------------#
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
        -new_image = next image in line to be processed
    Function output:
        -displacement values of the current frame compared to the previous eigenframe
        -It also updates the state.eigenframe value
    """
    new_image = state.preprocessed_image_array

    if analysis_mode == 'gpu':
        new_image = np_or_cp.array(new_image)
    else:
        pass
    if state.is_first_frame:
        #note: we initially defined the eframe as state.eigenframe[:,:] = np_or_cp.fft.fft2(new_image[:,:]),
        #but that resulted in state.eigenframe discarding the complex parts of the values.
        eigenframe_reciprocal_time = np_or_cp.reciprocal(new_image+state.offset)
        state.eigenframe_reciprocal_freq = np_or_cp.fft.fft2(eigenframe_reciprocal_time)
        # print("new_image                  = %.15e"%(new_image[489][489]))
        # print("shift                      = %.15e"%(state.offset))
        # print("eigenframe_reciprocal_time = %.15e"%(eigenframe_reciprocal_time[489][489]))
        # print("eigenframe_reciprocal_freq = %.15e %.15e"%(state.eigenframe_reciprocal_freq[489][489].real,state.eigenframe_reciprocal_freq[489][489].imag))
        state.is_first_frame = False
        vertical_displacement = state.v_0
        horizontal_displacement = state.h_0
    else: #perform the analysis
        #step 1. Prepare variables for shot noise weighting
        # print("new_image                  = %.15e"%(new_image[489][489]))
        new_image_offset = np_or_cp.add(new_image, state.offset)
        new_image_squared = np_or_cp.square(new_image_offset)
        new_image_squared_freq = np_or_cp.fft.fft2(new_image_squared)
        # print(f"new_image_offset : {new_image_offset[489][489]:.15e}")
        # print(f"new_image_squared: {new_image_squared[489][489]:.15e}")
        # print(f"new_image_sfreq  : {new_image_squared_freq[489][489].real:.15e} {new_image_squared_freq[489][489].imag:.15e}")
        #step 2. Correlate the previous Eframe and the new frame to obtain displacements-------------------------
        vertical_displacement, horizontal_displacement = phase_cross_correlation(state.eigenframe_reciprocal_freq, new_image_squared_freq, state.upsample_factor)
        #step 3. Shift the new frame onto the Eigenframe---------------------------------------------------------------------
        new_image_reciprocal = np_or_cp.reciprocal(new_image_offset)
        new_image_reciprocal_freq = np_or_cp.fft.fft2(new_image_reciprocal)
        new_image_reciprocal_freq_shifted = freq_shift(new_image_reciprocal_freq, horizontal_displacement, vertical_displacement, state.N_horizontal, state.N_vertical)
        #step 4. Build the next Eigenframe---------------------------------------------------------------------
        state.eigenframe_reciprocal_freq = state.A0*new_image_reciprocal_freq_shifted + state.B1*state.eigenframe_reciprocal_freq

    if analysis_mode == 'cpu':
        return [horizontal_displacement, vertical_displacement]
    else:
        return np_or_cp.asnumpy(np_or_cp.array([horizontal_displacement, vertical_displacement]))


def terminate_cross_correlation(state):
    pass
