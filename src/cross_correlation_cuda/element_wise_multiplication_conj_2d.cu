#include "image_analysis.h"

// GPU kernel
__global__
void element_wise_multiplication_conj_1d_gpu(
    const int n,
    const COMPLEX *restrict A,
    const COMPLEX *restrict B,
    COMPLEX *restrict C ) {
  /*
   *  Computes the element-wise multiplication C = A.B^{*},
   *  where B^{*} is the complex conjugate of B.
   *
   *  Inputs
   *  ------
   *    n : Size of the arrays.
   *    A : Complex array of size n.
   *    B : Complex array of size n.
   *    C : Complex array of size n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride)
    C[i] = CMUL(A[i],CONJ(B[i]));
}

// CPU wrapper
extern "C" __host__
void element_wise_multiplication_conj_2d(
    const int m,
    const int n,
    const COMPLEX *restrict A,
    const COMPLEX *restrict B,
    COMPLEX *restrict C ) {
  /*
   *  Computes the element-wise multiplication C = A.B^{*},
   *  where B^{*} is the complex conjugate of B, for all
   *  elements of two-dimensional (flattened) arrays.
   *
   *  Inputs
   *  ------
   *    m : Size of the first dimension of the array.
   *    n : Size of the second dimension of the array.
   *    A : Complex array of size m*n.
   *    B : Complex array of size m*n.
   *    C : Complex array of size m*n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */
  element_wise_multiplication_conj_GPU<<<MIN(n,512),MIN(m,512)>>>(m*n,A,B,C);
}