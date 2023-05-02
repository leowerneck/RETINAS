#include "image_analysis.h"

extern "C" __host__
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

  // Step 1: Free memory for all device arrays
  cudaFree(state->aux_array_int);
  cudaFree(state->aux_array_real);
  cudaFree(state->aux_array1);
  cudaFree(state->aux_array2);
  cudaFree(state->aux_array3);
  cudaFree(state->new_image_time);
  cudaFree(state->new_image_freq);
  cudaFree(state->reciprocal_new_image_time);
  cudaFree(state->ref_image_freq);
  cudaFree(state->image_sum_freq);

  // Step 2: Destroy FFT plans
  FFT_DESTROY_PLAN(state->fft2_plan);

  // Step 3: Destroy cuBLAS handle
  cublasDestroy(state->cublasHandle);

  // Step 4: Free memory allocated for the state struct
  free(state);

  info("Successfully finalized state object\n");
}