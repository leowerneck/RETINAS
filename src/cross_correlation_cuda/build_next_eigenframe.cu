#include "image_analysis.h"

__global__
void shift_image_add_to_eigenframe_GPU(const int N_horizontal,
                                       const int N_vertical,
                                       const REAL A0,
                                       const REAL B1,
                                       const COMPLEX *moving_frame_freq,
                                       const COMPLEX *reverse_shifts,
                                       COMPLEX *eigenframe_freq) {
  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int idx=tid;idx<N_horizontal*N_vertical;idx+=stride) {
    const COMPLEX product = CMUL(moving_frame_freq[idx],reverse_shifts[idx]);
    eigenframe_freq[idx].x = A0*product.x + B1*eigenframe_freq[idx].x;
    eigenframe_freq[idx].y = A0*product.y + B1*eigenframe_freq[idx].y;
  }
}

extern "C" __host__
void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    displacements : Array containing the horizontal and vertical displacements.
   *    state        : The C state object, containing the new eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Reverse shifts are stored in aux_array3
  shift_image_add_to_eigenframe_GPU<<<MIN(Nv,512),MIN(Nh,512)>>>(state->N_horizontal,
                                                                 state->N_vertical,
                                                                 state->A0,
                                                                 state->B1,
                                                                 state->new_image_freq_domain,
                                                                 state->aux_array3,
                                                                 state->eigenframe_freq_domain);
}