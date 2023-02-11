#include "image_analysis.h"

void set_zeroth_eigenframe( state_struct *restrict state ) {
  /*
   *  Initialize zeroth eigenframe to the FFT of the new image.
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
  //         store it as the zeroth eigenframe.
  FFTW_EXECUTE_DFT(state->fft2_plan,
                   state->new_image_time,
                   state->ref_image_freq);

  FFTW_EXECUTE_DFT_R2C(state->Itime, state->Efreq);
}
