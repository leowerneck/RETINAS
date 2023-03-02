#include "retina.h"

/*
 *  Function: get_eigenframe
 *  Author  : Leo Werneck
 *
 *  Returns the current eigenframe.
 *
 *  Arguments
 *  ---------
 *    state : in
 *      The state object (see retina.h).
 *
 *    eigenframe : out
 *      Real array that stores the eigenframe.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void get_eigenframe(
    state_struct *restrict state,
    REAL *restrict eigenframe ) {

  // Step 1: Compute the inverse FFT of the current eigenframe.
  FFTW_EXECUTE_DFT(state->ifft2_plan, state->ref_image_freq, state->aux_array1);

  // Step 2: The eigenframe should be real, so the imaginary components
  //         of aux_array1 should be tiny. We compute the eigenframe
  //         as the absolute value of aux_array1.
  const REAL norm = 1.0/(state->N_horizontal*state->N_vertical);
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i + state->N_horizontal * j;
      eigenframe[idx] = norm*CABS(state->aux_array1[idx]);
    }
  }
}
