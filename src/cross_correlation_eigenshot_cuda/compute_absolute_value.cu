#include "image_analysis.h"

// GPU kernel: Compute absolute value x = abs(z)
__global__
void compute_absolute_value_GPU(const int n,
                                const COMPLEX *restrict z,
                                REAL *restrict x) {
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride) {
    const COMPLEX zL = z[i];
    const REAL z_re  = zL.x;
    const REAL z_im  = zL.y;
    x[i] = z_re*z_re + z_im*z_im;
  }
}

// CPU wrapper: calls the function above appropriately
extern "C" __host__
void compute_absolute_value(const int m,
                            const int n,
                            const int mn,
                            const COMPLEX *restrict z,
                            REAL *restrict x) {
  compute_absolute_value_GPU<<<MIN(n,512),MIN(m,512)>>>(mn,z,x);
}