#include "image_analysis.h"

void set_zeroth_eigenframe( Cstate_struct *restrict Cstate ) {
  /*
   *  Initialize zeroth eigenframe to the FFT of the new image.
   *
   *  Inputs
   *  ------
   *    Cstate : The C state object, containing the new image.
   *
   *  Returns
   *  -------
   *     Nothing.
   */

  // Step 1: Compute FFT of the new_image_time_domain and
  //         store it as the zeroth eigenframe.
  FFTW_EXECUTE_DFT(Cstate->fft2_plan,
                   Cstate->new_image_time_domain,
                   Cstate->eigenframe_freq_domain);
}
