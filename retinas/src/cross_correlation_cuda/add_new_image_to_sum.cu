#include "image_analysis.h"

/*
 *  Construct the next reference image using:
 *   ref_image_new = A0*new_image_shifted + B1*ref_image_old.
 *
 *  Inputs
 *  ------
 *    N_horizontal    : Number of horizontal pixels in the image.
 *    N_vertical      : Number of vertical pixels in the image.
 *    A0              : IIR filter parameter (see eq. above).
 *    B1              : IIR filter parameter (see eq. above).
 *    new_image_freq  : FFT of the new image.
 *    reverse_shifts  : Reverse shift matrix.
 *    ref_image_freq  : FFT of the reference image. Stores the result.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
__global__
static void shift_new_image_add_to_image_sum_gpu(
    const int N_horizontal,
    const int N_vertical,
    const REAL A0,
    const REAL B1,
    const COMPLEX *new_image_freq,
    const COMPLEX *reverse_shifts,
    COMPLEX *image_sum_freq ) {

  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int i=tid;i<N_horizontal*N_vertical;i+=stride) {
    const COMPLEX product = CMUL(new_image_freq[i], reverse_shifts[i]);
    image_sum_freq[i].x += A0*product.x + B1*ref_image_freq[i].x;
    image_sum_freq[i].y += A0*product.y + B1*ref_image_freq[i].y;
  }
}

/*
 *  Function: add_new_image_to_sum
 *  Author  : Leo Werneck
 *
 *  Add the new image to the accumulator.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
extern "C" __host__
void add_new_image_to_sum(
    const REAL *restrict displacements,
    state_struct *restrict state ) {

  // Step 1: Compute the reverse shift matrix
  const int Nh = state->N_horizontal;
  const int Nv = state->N_vertical;
  compute_reverse_shift_matrix(Nh, Nv, displacements,
                               state->horizontal_shifts,
                               state->vertical_shifts,
                               state->shift_matrix);

  // Step 2: Now shift the new image and add it to the accumulator
  shift_new_image_add_to_image_sum_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
    Nh, Nv, state->A0, state->B1, state->new_image_freq,
    state->shift_matrix, state->image_sum_freq);

  state->image_counter++;
}
