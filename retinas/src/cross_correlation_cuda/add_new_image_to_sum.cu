#include "image_analysis.h"

// GPU kernel
__global__
static void add_new_image_to_sum_gpu(
    const int n,
    const COMPLEX *restrict new_image_freq,
    COMPLEX *restrict image_sum_freq ) {
  /*
   *  Compute the absolute value of all elements of an array.
   *
   *  Inputs
   *  ------
   *    n : Size of the arrays.
   *    z : Complex array of size n.
   *    x : Real array of size n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    image_sum_freq[i].x += new_image_freq[i].x;
    image_sum_freq[i].y += new_image_freq[i].y;
  }
}

extern "C" __host__
void add_new_image_to_sum( state_struct *restrict state ) {
  /*
   *  This is the CPU wrapper to the function above.
   */
  const int Nh = state->N_horizontal;
  const int Nv = state->N_vertical;
  add_new_image_to_sum_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
    state->NhNv, state->new_image_freq, state->image_sum_freq);
  state->image_counter++;
}