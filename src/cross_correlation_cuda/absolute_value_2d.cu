#include "image_analysis.h"

// GPU kernel
__global__
static void absolute_value_1d_gpu(
    const int n,
    const COMPLEX *restrict z,
    REAL *restrict x ) {
  /*
   *  Compute the absolute value of all elements of an array.
   *
   *  Inputs
   *  ------
   *    n : Size of the arrays.
   *    z : Complex array of size n.
   *    x : Real array of size n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    const COMPLEX zL = z[i];
    const REAL z_re  = zL.x;
    const REAL z_im  = zL.y;
    x[i] = z_re*z_re + z_im*z_im;
  }
}

extern "C" __host__
void absolute_value_2d(
    const int m,
    const int n,
    const COMPLEX *restrict z,
    REAL *restrict x ) {
  /*
   *  This is the CPU wrapper to the function above.
   */
  absolute_value_1d_gpu<<<MIN(n,512),MIN(m,512)>>>(m*n, z, x);
}