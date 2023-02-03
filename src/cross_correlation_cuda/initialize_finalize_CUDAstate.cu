#include "image_analysis.h"

extern "C" __host__
state_struct *initialize_state( const int N_horizontal,
                                  const int N_vertical,
                                  const REAL upsample_factor,
                                  const REAL A0,
                                  const REAL B1 ) {

  // Step 1: Allocate memory for the parameter struct
  state_struct *state = (state_struct *)malloc(sizeof(state_struct));

  // Step 2: Copy Python parameters to the C state struct
  state->N_horizontal    = N_horizontal;
  state->N_vertical      = N_vertical;
  state->upsample_factor = upsample_factor;
  state->A0              = A0;
  state->B1              = B1;

  // Step 3: Define auxiliary variables
  const int NhNv         = N_horizontal * N_vertical;
  const int N_upsampling = (int)CEIL(upsample_factor * 1.5);
  const int aux_size     = MAX(N_horizontal,N_upsampling)*MAX(N_vertical,N_upsampling);

  // Step 4: Allocate memory for host quantities
  state->host_aux_array = (COMPLEX *)malloc(sizeof(COMPLEX)*NhNv);

  // Step 5: Allocate memory for device quantities
  // Step 5.a: Auxiliary arrays
  cudaMalloc(&state->aux_array_int ,sizeof(uint16_t)*aux_size);
  cudaMalloc(&state->aux_array_real,sizeof(REAL)    *aux_size);
  cudaMalloc(&state->aux_array1    ,sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array2    ,sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array3    ,sizeof(COMPLEX) *aux_size);

  // Step 5.b: Arrays that hold the images
  cudaMalloc(&state->new_image_time_domain ,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->new_image_freq_domain ,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->eigenframe_freq_domain,sizeof(COMPLEX)*NhNv);

  // Step 5.c: Create the FFT plan
  cufftPlan2d(&state->fft2_plan,N_vertical,N_horizontal,FFT_C2C);

  // Step 5.d: Create the cuBLAS handle
  cublasCreate(&state->cublasHandle);

  // Step 6: All done! Return the CUDA state object
  return state;
}

extern "C" __host__
int finalize_state( state_struct *restrict state ) {

  // Step 1: Free memory for all host arrays
  free(state->host_aux_array);

  // Step 2: Free memory for all device arrays
  cudaFree(state->aux_array_int);
  cudaFree(state->aux_array_real);
  cudaFree(state->aux_array1);
  cudaFree(state->aux_array2);
  cudaFree(state->aux_array3);
  cudaFree(state->new_image_time_domain);
  cudaFree(state->new_image_freq_domain);
  cudaFree(state->eigenframe_freq_domain);

  // Step 3: Destroy FFT plans
  FFT_DESTROY_PLAN(state->fft2_plan);

  // Step 4: Destroy cuBLAS handle
  cublasDestroy(state->cublasHandle);

  // Step 5: Free memory allocated for the state struct
  free(state);

  // Step 6: All done!
  return 0;

}
