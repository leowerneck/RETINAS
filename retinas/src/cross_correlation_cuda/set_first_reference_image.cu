#include "image_analysis.h"

__host__
void set_first_reference_image( state_struct *restrict state ) {
  /*
   *  Initialize first reference image to the FFT of the new image.
   *
   *  Inputs
   *  ------
   *    state : The C state object, containing the new image.
   *
   *  Returns
   *  -------
   *     Nothing.
   */

  // Step 1: Compute FFT of the new_image_time and
  //         store it as the first reference image.
  if( state->shot_noise_method )
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->reciprocal_new_image_time,
                    state->ref_image_freq,
                    CUFFT_FORWARD);
  else
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->new_image_time,
                    state->ref_image_freq,
                    CUFFT_FORWARD);
}