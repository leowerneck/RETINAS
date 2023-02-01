#include "image_analysis.h"

__global__
void typecast_shift_compute_reciprocal_and_copy_gpu(const int n, const uint16_t *input_array, REAL *real_array, COMPLEX *complex_array) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    const REAL z_real = (REAL)input_array[i];
    real_array[i]     = z_real;
    complex_array[i]  = MAKE_COMPLEX(z_real,0.0);
  }
}

extern "C" __host__
REAL typecast_and_return_brightness( const uint16_t *restrict input_array,
                                     CUDAstate_struct *restrict CUDAstate ) {

  // Step 1: Set useful constants
  const int Nh   = CUDAstate->N_horizontal;
  const int Nv   = CUDAstate->N_vertical;
  const int NhNv = Nh*Nv;

  // Step 2: Copy raw image from host (CPU) to device (GPU)
  cudaMemcpy(CUDAstate->aux_array_int,input_array,sizeof(uint16_t)*NhNv,cudaMemcpyHostToDevice);

  // Step 3: Typecast, shift, and compute the reciprocal of the input image.
  //         We also copy the image to a real array, allowing us to use a
  //         cuBLAS functino to compute the brightness in Step 4 below.
  typecast_shift_compute_reciprocal_and_copy_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(NhNv,
                                                                              CUDAstate->aux_array_int,
                                                                              CUDAstate->aux_array_real,
                                                                              CUDAstate->new_image_time_domain);

  // Step 4: Compute the brightness
  REAL brightness;
  CUBLASASUM(CUDAstate->cublasHandle, NhNv, CUDAstate->aux_array_real, 1, &brightness);

  // Step 5: All done! Return the brightness
  return brightness;
}
