#include "image_analysis.h"

void state_finalize( state_struct *restrict state ) {
  /*
   *  Free memory for all pointers in the C state object,
   *  as well as the memory allocated for the object.
   *
   *  Inputs
   *  ------
   *    state : Pointer to the state object.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Free memory for all arrays
  FFTW_FREE(state->aux_array1);
  FFTW_FREE(state->aux_array2);
  FFTW_FREE(state->aux_array3);
  FFTW_FREE(state->new_image_time);
  FFTW_FREE(state->new_image_freq);
  FFTW_FREE(state->eigenframe_freq);

  // Step 2: Destroy FFT plans
  FFTW_DESTROY_PLAN(state->fft2_plan);
  FFTW_DESTROY_PLAN(state->ifft2_plan);

  // Step 3: Free memory allocated for the C state struct
  free(state);

  info("Successfully finalized C state object\n");
}
