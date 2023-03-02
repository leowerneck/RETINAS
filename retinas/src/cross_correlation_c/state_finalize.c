#include "retinas.h"

/*
 *  Function: state_finalize
 *  Author  : Leo Werneck
 *
 *  Free memory all memory used by the program.
 *
 *  Arguments
 *  ------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void state_finalize( state_struct *restrict state ) {

  // Step 1: Free memory for all arrays
  FFTW_FREE(state->aux_array1);
  FFTW_FREE(state->aux_array2);
  FFTW_FREE(state->aux_array3);
  FFTW_FREE(state->new_image_time);
  FFTW_FREE(state->new_image_freq);
  FFTW_FREE(state->ref_image_freq);

  // Step 2: Destroy FFT plans
  FFTW_DESTROY_PLAN(state->fft2_plan);
  FFTW_DESTROY_PLAN(state->ifft2_plan);

  // Step 3: Free memory allocated for the C state struct
  free(state);

  info("Successfully finalized state object\n");
}
