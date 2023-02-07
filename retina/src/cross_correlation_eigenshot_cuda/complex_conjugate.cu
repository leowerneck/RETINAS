#include "image_analysis.h"

__global__
void complex_conjugate_GPU(const int n, COMPLEX *restrict z) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride)
    z[i].y = -z[i].y;
}

extern "C" __host__
void complex_conjugate(const int m, const int n, const int mn, COMPLEX *restrict z) {
  complex_conjugate_GPU<<<MIN(n,512),MIN(m,512)>>>(mn,z);
}