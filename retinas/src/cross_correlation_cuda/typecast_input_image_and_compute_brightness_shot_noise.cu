#include "image_analysis.h"

__global__
static void typecast_shift_square_reciprocal_and_copy_1d_gpu(
    const int n,
    const REAL shift,
    const uint16_t *restrict input_array,
    REAL *restrict real_array,
    COMPLEX *restrict reciprocal_image,
    COMPLEX *restrict squared_image ) {
  /*
   *  Typecast the input image from uint16 to REAL; copy into complex array.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      n             : Array size.
   *      shift         : Added to the image before taking the reciprocal.
   *      input_array   : Unsigned int16 array size n.
   *
   *    Outputs
   *    -------
   *      real_array       : Copy of the input image converted to REAL.
   *      reciprocal_image : Reciprocal of input image as COMPLEX.
   *      squared_image    : Square of the input image as COMPLEX.
   *
   *  Returns
   *  -------
   *     Nothing.
   */

  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    const REAL z_real   = (REAL)input_array[i];
    const REAL z_shift  = z_real + shift;
    real_array[i]       = z_real;
    reciprocal_image[i] = MAKE_COMPLEX(1.0/z_shift,0.0);
    squared_image[i]    = MAKE_COMPLEX(z_shift*z_shift,0.0);
  }
}

extern "C" __host__
REAL typecast_input_image_and_compute_brightness_shot_noise(
    const uint16_t *restrict input_array,
    state_struct *restrict state ) {
  /*
   *  Typecast the input image from uint16 to REAL and COMPLEX. Also
   *  compute the brightness, which is the sum of all pixel values in
   *  the image. This function uses cuBLAS to compute the brightness.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      input_array : Input image stored as a 1D array.
   *
   *    Outputs
   *    -------
   *      state : State object; results stored in state->aux_array_real,
   *              state->new_image_time (actually the reciprocal),
   *              and state->new_image_time_squared.
   *
   *  Returns
   *  -------
   *     brightness : Brightness of the image.
   */

  // Step 1: Set useful constants
  const int Nh   = state->N_horizontal;
  const int Nv   = state->N_vertical;
  const int NhNv = Nh*Nv;

  // Step 2: Copy raw image from host (CPU) to device (GPU)
  cudaMemcpy(state->aux_array_int, input_array, sizeof(uint16_t)*NhNv, cudaMemcpyHostToDevice);

  // Step 3: Typecast, shift, and compute the reciprocal of the input image.
  //         We also copy the image to a real array, allowing us to use a
  //         cuBLAS functino to compute the brightness in Step 4 below.
  typecast_shift_square_reciprocal_and_copy_1d_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
      NhNv,
      state->shift,
      state->aux_array_int,
      state->aux_array_real,
      state->reciprocal_new_image_time,
      state->new_image_time);

  // Step 4: Compute the brightness
  REAL brightness;
  CUBLASASUM(state->cublasHandle, NhNv, state->aux_array_real, 1, &brightness);

  // Step 5: All done! Return the brightness
  return brightness;
}
