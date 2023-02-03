#include "image_analysis.h"

// This function is required by the upsampling algorithm.
// It returns the frequency bins of a DFT. The expected
// behavior is the following:
// For a given size n and sample spacing d, the expected
// output is:
// [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n),
// if n is even, and
// [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)
// if n is odd.
// This function was based on and validated against the
// NumPy function fft.fftfreq.
__global__
void fftfreq_gpu(const int n,const REAL d,REAL *restrict freq) {
  const int nhalf   = FLOOR((n-1)/2.0)+1;
  const int nmnhalf = n-nhalf;
  const REAL norm   = 1.0/(n*d);
  const int index   = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride  = blockDim.x * gridDim.x;
  for(int i=index;i<nhalf;i+=stride)
    freq[i] = i * norm;
  for(int i=index;i<nmnhalf;i+=stride)
    freq[i+nhalf] = (-FLOOR(n/2.0)+i) * norm;
}

// Optimized for n even
__global__
void fftfreq_gpu_opt(const int n,const REAL d,REAL *restrict freq) {
  const int nhalf  = n/2;
  const REAL norm  = 1.0/(n*d);
  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<nhalf;i+=stride) {
    freq[i      ]  = i * norm;
    freq[i+nhalf]  = (i-nhalf) * norm;
  }
}

// Wrapper function to fftfreq_gpu
__host__
void fftfreq(const int n, const REAL d, REAL *freq) {
  if( n%2 == 0 ) fftfreq_gpu_opt<<<MIN(n/4,512),MIN(n/4,512)>>>(n,d,freq);
  else           fftfreq_gpu<<<MIN(n/4,512),MIN(n/4,512)>>>(n,d,freq);
}

__global__
void compute_horizontal_kernel_gpu(const int S, const int Nh, const REAL offset_h, const REAL *restrict freqs_h, COMPLEX *restrict kernel_h) {
  int tids    = threadIdx.x + blockIdx.x*blockDim.x;
  int tidx    = threadIdx.y + blockIdx.y*blockDim.y;
  int strides = blockDim.x*gridDim.x;
  int stridex = blockDim.y*gridDim.y;
  COMPLEX im2pi = MAKE_COMPLEX(0.0,-2.0*M_PI);
  for(int i_h=tidx;i_h<Nh;i_h+=stridex) {
    for(int i_s=tids;i_s<S;i_s+=strides) {
      const COMPLEX kernel_l = MAKE_COMPLEX((i_s - offset_h)*freqs_h[i_h],0.0);
      kernel_h[i_s+S*i_h]    = CEXP(CMUL(im2pi,kernel_l));
    }
  }
}

__global__
void compute_vertical_kernel_gpu(const int Nv, const int S, const REAL offset_v, const REAL *restrict freqs_v, COMPLEX *restrict kernel_v) {
  int tidy    = threadIdx.x + blockIdx.x*blockDim.x;
  int tids    = threadIdx.y + blockIdx.y*blockDim.y;
  int stridey = blockDim.x*gridDim.x;
  int strides = blockDim.y*gridDim.y;
  COMPLEX im2pi = MAKE_COMPLEX(0.0,-2.0*M_PI);
  for(int i_s=tids;i_s<S;i_s+=strides) {
    for(int i_v=tidy;i_v<Nv;i_v+=stridey) {
      const COMPLEX kernel_l = MAKE_COMPLEX((i_s - offset_v)*freqs_v[i_v],0.0);
      kernel_v[i_v+Nv*i_s]   = CEXP(CMUL(im2pi,kernel_l));
    }
  }
}

extern "C" __host__
void compute_horizontal_kernel(const REAL *restrict sample_region_offset, state_struct *restrict state) {
  const int Nh = state->N_horizontal;
  const int S  = (int)(state->upsample_factor*1.5+0.5);
  // aux_array1 contains the image product at this point. Use 2 or 3 instead.
  // Compute the horizontal frequencies and store them in freq_array
  fftfreq(Nh, state->upsample_factor, state->aux_array_real);
  // Now compute the horizontal kernel, storing it in aux_array2
  compute_horizontal_kernel_gpu<<<dim3(32,16),dim3(32,16)>>>(S,Nh,sample_region_offset[0],state->aux_array_real,state->aux_array2);
}

extern "C" __host__
void compute_vertical_kernel(const REAL *restrict sample_region_offset, state_struct *restrict state) {
  const int Nv = state->N_vertical;
  const int S  = (int)(state->upsample_factor*1.5+0.5);
  // aux_array3 contains the contraction at this point. Use 1 or 2 instead.
  // Compute the vertical frequencies and store them in freq_array
  fftfreq(Nv, state->upsample_factor, state->aux_array_real);
  // Now compute the vertical kernel, storing it in aux_array2
  compute_vertical_kernel_gpu<<<dim3(32,16),dim3(32,16)>>>(Nv,S,sample_region_offset[1],state->aux_array_real,state->aux_array2);
}