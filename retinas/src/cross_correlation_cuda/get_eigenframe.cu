#include "image_analysis.h"

__host__
void get_eigenframe(
    state_struct *restrict state,
    REAL *restrict eigenframe_time ) {
  /*
   *  Returns the current eigenframe.
   *
   *  Inputs
   *  ------
   *    state      : The state object (see image_analysis.h).
   *    eigenframe : Real array that stores the eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the inverse FFT of the current eigenframe.
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->eigenframe_freq,
                  state->eigenframe_time,
                  CUFFT_INVERSE);

  // Step 2: The eigenframe should be real, so the imaginary components
  //         of aux_array1 should be tiny. We compute the eigenframe
  //         as the absolute value of aux_array1.
  absolute_value_2d(state->N_horizontal,
                    state->N_vertical,
                    state->eigenframe_time,
                    state->aux_array_real);

  // Step 3: Now copy the data from the GPU to the CPU
  const int NhNv = state->N_horizontal * state->N_vertical;
  cudaMemcpy(eigenframe_time,
             state->aux_array_real,
             sizeof(REAL)*NhNv,
             cudaMemcpyDeviceToHost);
}
