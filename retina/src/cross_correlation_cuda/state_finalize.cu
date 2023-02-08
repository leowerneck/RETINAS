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

  // Step 1: Free memory for all host arrays
  free(state->host_aux_array);

  // Step 2: Free memory for all device arrays
  cudaFree(state->aux_array_int);
  cudaFree(state->aux_array_real);
  cudaFree(state->aux_array1);
  cudaFree(state->aux_array2);
  cudaFree(state->aux_array3);
  cudaFree(state->new_image_time);
  cudaFree(state->new_image_freq);
  cudaFree(state->new_image_time_squared);
  cudaFree(state->eigenframe_freq);

  // Step 3: Destroy FFT plans
  FFT_DESTROY_PLAN(state->fft2_plan);

  // Step 4: Destroy cuBLAS handle
  cublasDestroy(state->cublasHandle);

  // Step 5: Free memory allocated for the state struct
  free(state);

  info("Successfully finalized state object\n");
}