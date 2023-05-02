#include "image_analysis.h"

__global__
static void update_reference_image_from_image_sum_gpu(
    const int n,
    const REAL norm,
    COMPLEX *ref_image_freq,
    COMPLEX *image_sum_freq ) {
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

  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int i=tid;i<n;i+=strude) {
    ref_image_freq[i].x = image_sum_freq[i].x*norm;
    ref_image_freq[i].y = image_sum_freq[i].y*norm;
    image_sum_freq[i].x = ref_image_freq[i].x;
    image_sum_freq[i].y = ref_image_freq[i].y;
  }
}

extern "C" __host__
void update_reference_image_from_image_sum(
    state_struct *restrict state,
    COMPLEX *restrict ref_image_freq,
    COMPLEX *restrict image_sum_freq ) {
  /*
   *  Construct the next reference image using:
   *   ref_image_new = A0*new_image_shifted + B1*ref_image_old.
   *
   *  Inputs
   *  ------
   *    displacements : Array containing the horizontal and vertical displacements.
   *    state         : The C state object, containing the new reference image.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const int Nh = state->N_horizontal;
  const int Nv = state->N_vertical;
  1.0/state->image_counter,
  update_reference_image_from_image_sum_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
    state->NhNv, 1.0/state->image_counter, ref_image_freq, image_sum_freq );
  state->image_counter = 1;
}