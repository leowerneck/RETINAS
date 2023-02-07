#include "image_analysis.h"

void get_eigenframe(
    state_struct *restrict state,
    REAL *restrict eigenframe ) {
  /*
   *  Returns the current eigenframe.
   *
   *  Inputs
   *  ------
   *    state      : The C state object.
   *    eigenframe : Real array that stores the eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the inverse FFT of the current eigenframe.
  FFTW_EXECUTE_DFT(state->ifft2_plan, state->eigenframe_freq_domain, state->aux_array1);

  // Step 2: The eigenframe should be real, so the imaginary components
  //         of aux_array1 should be tiny. We compute the eigenframe
  //         as the absolute value of aux_array1.
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i + state->N_horizontal * j;
      eigenframe[idx] = CABS(state->aux_array1[idx]);
    }
  }
}
