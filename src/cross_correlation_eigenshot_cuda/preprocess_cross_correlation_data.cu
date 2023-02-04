#include "image_analysis.h"

__global__
void typecast_shift_square_reciprocal_and_copy_gpu(
    const int n,
    const REAL shift,
    const uint16_t *input_array,
    REAL *real_array,
    COMPLEX *reciprocal_image,
    COMPLEX *squared_image) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    const REAL z_real   = (REAL)input_array[i];
    const REAL z_shift  = z_real + shift;
    real_array[i]       = z_real;
    reciprocal_image[i] = MAKE_COMPLEX(1.0/z_shift,0.0);
    squared_image[i]    = MAKE_COMPLEX(z_shift*z_shift,0.0);
  }
}

__global__
void typecast_rebin_4x4_shift_square_reciprocal_and_copy_gpu( const int N_horizontal,
                                                              const int N_vertical,
                                                              const REAL shift,
                                                              const uint16_t *restrict input_array,
                                                              REAL *restrict real_array,
                                                              COMPLEX *reciprocal_image,
                                                              COMPLEX *squared_image ) {

  const int tidx    = 4*(threadIdx.x + blockIdx.x * blockDim.x);
  const int tidy    = 4*(threadIdx.y + blockIdx.y * blockDim.y);
  const int stridex = 4*(blockDim.x * gridDim.x);
  const int stridey = 4*(blockDim.y * gridDim.y);

  const int N_horizontal_original = 4*N_horizontal;
  const int N_vertical_original   = 4*N_horizontal;
  for(int j=tidy;j<N_vertical_original;j+=stridey) {
    for(int i=tidx;i<N_horizontal_original;i+=stridex) {
      REAL bin_value = 0.0;
      for(int jj=0;jj<4;jj++) {
        for(int ii=0;ii<4;ii++) {
          bin_value += (REAL)input_array[(i+ii) + N_horizontal_original*(j+jj)];
        }
      }
      const int index         = (i + N_horizontal*j)/4;
      const REAL z_shift      = bin_value + shift;
      real_array[index]       = bin_value;
      reciprocal_image[index] = MAKE_COMPLEX(1.0/z_shift,0.0);
      squared_image[index]    = MAKE_COMPLEX(z_shift*z_shift,0.0);
    }
  }
}

extern "C" __host__
REAL typecast_input_image_and_compute_brightness(
    const uint16_t *restrict input_array,
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
  typecast_shift_square_reciprocal_and_copy_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(NhNv,
                                                                             CUDAstate->shift,
                                                                             CUDAstate->aux_array_int,
                                                                             CUDAstate->aux_array_real,
                                                                             CUDAstate->reciprocal_new_image_time,
                                                                             CUDAstate->squared_new_image_time);

  // Step 4: Compute the brightness
  REAL brightness;
  CUBLASASUM(CUDAstate->cublasHandle, NhNv, CUDAstate->aux_array_real, 1, &brightness);

  // Step 5: All done! Return the brightness
  return brightness;
}

extern "C" __host__
REAL typecast_rebin_4x4_and_return_brightness( const uint16_t *restrict input_array,
                                               CUDAstate_struct *restrict CUDAstate ) {

  // Step 1: Set useful constants
  const int Nh   = CUDAstate->N_horizontal;
  const int Nv   = CUDAstate->N_vertical;
  const int NhNv = 16*Nh*Nv;

  // Step 2: Copy raw image from host (CPU) to device (GPU)
  cudaMemcpy(CUDAstate->aux_array_int,input_array,sizeof(uint16_t)*NhNv,cudaMemcpyHostToDevice);

  // Step 3: Rebin and typecast
  typecast_rebin_4x4_shift_square_reciprocal_and_copy_gpu<<<dim3(32,16),dim3(32,16)>>>( Nh, Nv,
                                                                                        CUDAstate->shift,
                                                                                        CUDAstate->aux_array_int,
                                                                                        CUDAstate->aux_array_real,
                                                                                        CUDAstate->reciprocal_new_image_time,
                                                                                        CUDAstate->squared_new_image_time );

  // Step 4: Compute the brightness
  REAL brightness;
  CUBLASASUM(CUDAstate->cublasHandle, NhNv, CUDAstate->aux_array_real, 1, &brightness);

  // Step 5: All done! Return the brightness
  return brightness;
}