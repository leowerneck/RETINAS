#include "retinas.h"

__global__
static void update_reference_image_from_image_sum_gpu(
    const int N_horizontal,
    const int N_vertical,
    const REAL norm,
    COMPLEX *image_sum_freq,
    COMPLEX *ref_image_freq ) {

  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int i=tid;i<N_horizontal*N_vertical;i+=stride) {
    ref_image_freq[i].x = image_sum_freq[i].x * norm;
    ref_image_freq[i].y = image_sum_freq[i].y * norm;
    image_sum_freq[i].x = image_sum_freq[i].y = 0.0;
  }
}

extern "C" __host__
void update_reference_image_from_image_sum( state_struct *restrict state ) {
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

  // Step 1: Compute the reverse shift matrix
  const int Nh    = state->N_horizontal;
  const int Nv    = state->N_vertical;
  const REAL norm = 1.0/state->image_counter;
  update_reference_image_from_image_sum_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
    Nh, Nv, norm, state->image_sum_freq, state->ref_image_freq);

  state->image_counter = 0;
}
