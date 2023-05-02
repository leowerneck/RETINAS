#include "image_analysis.h"

__host__
void get_reference_image_freq(
    state_struct *restrict state,
    COMPLEX *restrict ref_image_freq ) {
  /*
   *  Returns the FFT of the current reference image.
   *
   *  Inputs
   *  ------
   *    state          : The state object (see image_analysis.h).
   *    ref_image_freq : Complex array that stores the FFT of the reference image.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const int NhNv = state->N_horizontal * state->N_vertical;
  cudaMemcpy(ref_image_freq,
             state->ref_image_freq,
             sizeof(REAL)*NhNv,
             cudaMemcpyDeviceToHost);
}
