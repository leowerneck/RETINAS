#include "image_analysis.h"

extern "C" __host__
CUDAstate_struct *initialize_CUDAstate( const int N_horizontal,
                                        const int N_vertical,
                                        const REAL upsample_factor,
                                        const REAL A0,
                                        const REAL B1,
                                        const REAL shift ) {

  // Step 1: Allocate memory for the parameter struct
  CUDAstate_struct *CUDAstate = (CUDAstate_struct *)malloc(sizeof(CUDAstate_struct));

  // Step 2: Copy Python parameters to the C state struct
  CUDAstate->N_horizontal    = N_horizontal;
  CUDAstate->N_vertical      = N_vertical;
  CUDAstate->upsample_factor = upsample_factor;
  CUDAstate->A0              = A0;
  CUDAstate->B1              = B1;
  CUDAstate->shift           = shift;

  // Step 3: Define auxiliary variables
  const int NhNv         = N_horizontal * N_vertical;
  const int N_upsampling = (int)CEIL(upsample_factor * 1.5);
  const int aux_size     = MAX(N_horizontal,N_upsampling)*MAX(N_vertical,N_upsampling);

  // Step 4: Allocate memory for host quantities
  CUDAstate->host_aux_array = (COMPLEX *)malloc(sizeof(COMPLEX)*NhNv);

  // Step 5: Allocate memory for device quantities
  // Step 5.a: Auxiliary arrays
  cudaMalloc(&CUDAstate->aux_array_int ,sizeof(uint16_t)*aux_size);
  cudaMalloc(&CUDAstate->aux_array_real,sizeof(REAL)    *aux_size);
  cudaMalloc(&CUDAstate->aux_array1    ,sizeof(COMPLEX) *aux_size);
  cudaMalloc(&CUDAstate->aux_array2    ,sizeof(COMPLEX) *aux_size);
  cudaMalloc(&CUDAstate->aux_array3    ,sizeof(COMPLEX) *aux_size);

  // Step 5.b: Arrays that hold the images
  cudaMalloc(&CUDAstate->reciprocal_new_image_time ,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&CUDAstate->reciprocal_new_image_freq ,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&CUDAstate->reciprocal_eigenframe_freq,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&CUDAstate->squared_new_image_time    ,sizeof(COMPLEX)*NhNv);
  cudaMalloc(&CUDAstate->squared_new_image_freq    ,sizeof(COMPLEX)*NhNv);

  // Step 5.c: Create the FFT plan
  cufftPlan2d(&CUDAstate->fft2_plan,N_vertical,N_horizontal,FFT_C2C);

  // Step 5.d: Create the cuBLAS handle
  cublasCreate(&CUDAstate->cublasHandle);

  // Step 6: All done! Return the CUDA state object
  return CUDAstate;
}

extern "C" __host__
int finalize_CUDAstate( CUDAstate_struct *restrict CUDAstate ) {

  // Step 1: Free memory for all host arrays
  free(CUDAstate->host_aux_array);

  // Step 2: Free memory for all device arrays
  cudaFree(CUDAstate->aux_array_int);
  cudaFree(CUDAstate->aux_array_real);
  cudaFree(CUDAstate->aux_array1);
  cudaFree(CUDAstate->aux_array2);
  cudaFree(CUDAstate->aux_array3);
  cudaFree(CUDAstate->reciprocal_new_image_time);
  cudaFree(CUDAstate->reciprocal_new_image_freq);
  cudaFree(CUDAstate->reciprocal_eigenframe_freq);
  cudaFree(CUDAstate->squared_new_image_time);
  cudaFree(CUDAstate->squared_new_image_freq);

  // Step 3: Destroy FFT plans
  FFT_DESTROY_PLAN(CUDAstate->fft2_plan);

  // Step 4: Destroy cuBLAS handle
  cublasDestroy(CUDAstate->cublasHandle);

  // Step 5: Free memory allocated for the state struct
  free(CUDAstate);

  // Step 6: All done!
  return 0;

}
