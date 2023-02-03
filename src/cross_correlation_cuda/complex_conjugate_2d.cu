#include "image_analysis.h"

// GPU kernel
__global__
void complex_conjugate_1d_gpu(
    const int n,
    COMPLEX *restrict z ) {
  /*
   *  Compute the complex conjugate of all elements of an array.
   *
   *  Inputs
   *  ------
   *    n : Size of the array.
   *    z : Complex array of size n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride)
    z[i].y = -z[i].y;
}

// CPU wrapper
extern "C" __host__
void complex_conjugate_2d(
    const int m,
    const int n,
    COMPLEX *restrict z ) {
  /*
   *  Compute the complex conjugate of all elements of a
   *  two-dimensional (flattened) array.
   *
   *  Inputs
   *  ------
   *    m : Size of the first dimension of the array.
   *    n : Size of the second dimension of the array.
   *    z : Complex array of size m*n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */
  complex_conjugate_1d_gpu<<<MIN(n,512),MIN(m,512)>>>(m*n, z);
}