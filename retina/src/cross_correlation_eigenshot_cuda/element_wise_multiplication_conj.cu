#include "image_analysis.h"

// GPU kernel: Compute element wise multiplication C = A*B^{*}
__global__
void element_wise_multiplication_conj_GPU(const int n,
                                          const COMPLEX *restrict A,
                                          const COMPLEX *restrict B,
                                          COMPLEX *restrict C) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride)
    C[i] = CMUL(A[i],CONJ(B[i]));
}

// CPU wrapper: calls the function above appropriately
extern "C" __host__
void element_wise_multiplication_conj(const int m,
                                      const int n,
                                      const int mn,
                                      const COMPLEX *restrict A,
                                      const COMPLEX *restrict B,
                                      COMPLEX *restrict C) {
  element_wise_multiplication_conj_GPU<<<MIN(n,512),MIN(m,512)>>>(mn,A,B,C);
}