#include "image_analysis.h"

__global__
static void shift_image_add_to_eigenframe_gpu(
    const int N_horizontal,
    const int N_vertical,
    const REAL A0,
    const REAL B1,
    const COMPLEX *new_image_freq,
    const COMPLEX *reverse_shifts,
    COMPLEX *eigenframe_freq ) {
  /*
   *  Construct the next eigenframe using:
   *   eigenframe_new = A0*new_image_shifted + B1*eigenframe_old.
   *
   *  Inputs
   *  ------
   *    N_horizontal    : Number of horizontal pixels in the image.
   *    N_vertical      : Number of vertical pixels in the image.
   *    A0              : IIR filter parameter (see eq. above).
   *    B1              : IIR filter parameter (see eq. above).
   *    new_image_freq  : FFT of the new image.
   *    reverse_shifts  : Reverse shift matrix.
   *    eigenframe_freq : FFT of the eigenframe. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int i=tid;i<N_horizontal*N_vertical;i+=stride) {
    const COMPLEX product = CMUL(new_image_freq[i],reverse_shifts[i]);
    eigenframe_freq[i].x  = A0*product.x + B1*eigenframe_freq[i].x;
    eigenframe_freq[i].y  = A0*product.y + B1*eigenframe_freq[i].y;
  }
}

extern "C" __host__
void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state ) {
  /*
   *  Construct the next eigenframe using:
   *   eigenframe_new = A0*new_image_shifted + B1*eigenframe_old.
   *
   *  Inputs
   *  ------
   *    displacements : Array containing the horizontal and vertical displacements.
   *    state         : The C state object, containing the new eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the reverse shift matrix
  const int Nh = state->N_horizontal;
  const int Nv = state->N_vertical;
  compute_reverse_shift_matrix(Nh, Nv, displacements,
                               state->aux_array1,
                               state->aux_array2,
                               state->aux_array3);

  // Step 2: Now shift the new image and add it to the eigenframe
  // Note: in the shot noise method, the following identification is made:
  //   state->new_image_freq  == state->reciprocal_new_image_freq
  //   state->eigenframe_freq == state->reciprocal_eigenframe_freq
  shift_image_add_to_eigenframe_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
    Nh, Nv, state->A0, state->B1, state->new_image_freq,
    state->aux_array3, state->eigenframe_freq);
}