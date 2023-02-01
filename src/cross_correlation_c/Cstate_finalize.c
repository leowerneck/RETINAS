#include "image_analysis.h"

int Cstate_finalize( Cstate_struct *restrict Cstate ) {
  /*
   *  Free memory for all pointers in the C state object,
   *  as well as the memory allocated for the object.
   *
   *  Inputs
   *  ------
   *    Cstate : Cstate_struct *
   *      Pointer to the Cstate object.
   *
   *  Returns
   *  -------
   *    error_key : int
   *      0 - success
   */

  // Step 1: Free memory for all arrays
  FFTW_FREE(Cstate->aux_array1);
  FFTW_FREE(Cstate->aux_array2);
  FFTW_FREE(Cstate->aux_array3);
  FFTW_FREE(Cstate->new_image_time_domain);
  FFTW_FREE(Cstate->new_image_freq_domain);
  FFTW_FREE(Cstate->eigenframe_freq_domain);

  // Step 2: Destroy FFT plans
  FFTW_DESTROY_PLAN(Cstate->fft2_plan);
  FFTW_DESTROY_PLAN(Cstate->ifft2_plan);

  // Step 3: Free memory allocated for the C state struct
  free(Cstate);

  // Step 4: All done!
  return 0;

}
