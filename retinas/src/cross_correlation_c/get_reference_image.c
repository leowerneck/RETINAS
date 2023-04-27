#include "retinas.h"

/*
 *  Function: get_reference_image
 *  Author  : Leo Werneck
 *
 *  Returns the current reference image.
 *
 *  Arguments
 *  ---------
 *    state : in
 *      The state object (see retinas.h).
 *
 *    reference_image : out
 *      Real array that stores the reference image.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void get_reference_image(
    state_struct *restrict state,
    REAL *restrict reference_image ) {

  // Step 1: Compute the inverse FFT of the current reference image.
  FFTW_EXECUTE_DFT(state->ifft2_plan, state->ref_image_freq, state->aux_array1);

  // Step 2: The reference image should be real, so the imaginary components
  //         of aux_array1 should be tiny. We compute the reference image
  //         as the absolute value of aux_array1.
  const REAL norm = 1.0/(state->N_horizontal*state->N_vertical);
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i + state->N_horizontal * j;
      reference_image[idx] = norm*CABS(state->aux_array1[idx]);
    }
  }
}
