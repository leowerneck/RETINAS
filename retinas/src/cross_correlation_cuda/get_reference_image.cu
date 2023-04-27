#include "image_analysis.h"

__host__
void get_reference_image(
    state_struct *restrict state,
    REAL *restrict ref_image_time ) {
  /*
   *  Returns the current reference image.
   *
   *  Inputs
   *  ------
   *    state          : The state object (see image_analysis.h).
   *    ref_image_time : Real array that stores the reference image.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the inverse FFT of the current reference image.
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->ref_image_freq,
                  state->ref_image_time,
                  CUFFT_INVERSE);

  // Step 2: The reference image should be real, so the imaginary components
  //         of aux_array1 should be tiny. We compute the reference image
  //         as the absolute value of aux_array1.
  absolute_value_2d(state->N_horizontal,
                    state->N_vertical,
                    state->ref_image_time,
                    state->aux_array_real);

  // Step 3: Now copy the data from the GPU to the CPU
  const int NhNv = state->N_horizontal * state->N_vertical;
  cudaMemcpy(ref_image_time,
             state->aux_array_real,
             sizeof(REAL)*NhNv,
             cudaMemcpyDeviceToHost);
}
