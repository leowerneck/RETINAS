#include "image_analysis.h"

int set_zeroth_eigenframe( Cstate_struct *restrict Cstate ) {
  /*
   *  Initialize zeroth eigenframe to the FFT of the new image.
   *
   *  Inputs
   *  ------
   *    Cstate : Cstate_struct *
   *       The C state object, containing the new image.
   *
   *  Returns
   *  -------
   *     error_key : int
   *       0 - success
   */

  // Step 1: Compute FFT of the new_image_time_domain and
  //         store it as the zeroth eigenframe.
  FFTW_EXECUTE_DFT(Cstate->fft2_plan,Cstate->new_image_time_domain,Cstate->eigenframe_freq_domain);

  // Step 2: All done!
  return 0;

}
