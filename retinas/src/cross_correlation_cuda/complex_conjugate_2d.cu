#include "retinas.h"

__global__
static void complex_conjugate_1d_gpu(
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

extern "C" __host__
void complex_conjugate_2d(
    const int m,
    const int n,
    COMPLEX *restrict z ) {
  /*
   *  This is the CPU wrapper to the function above.
   */
  complex_conjugate_1d_gpu<<<MIN(n,512),MIN(m,512)>>>(m*n, z);
}