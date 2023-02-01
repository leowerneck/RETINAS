#include "image_analysis.h"

__global__
void shift_image_add_to_eigenframe_GPU(const int N_horizontal,
                                       const int N_vertical,
                                       const REAL A0,
                                       const REAL B1,
                                       const COMPLEX *moving_frame_freq,
                                       const COMPLEX *reverse_shifts,
                                       COMPLEX *eigenframe_freq) {
  int tid    = threadIdx.x + blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  for(int idx=tid;idx<N_horizontal*N_vertical;idx+=stride) {
    const COMPLEX product = CMUL(moving_frame_freq[idx],reverse_shifts[idx]);
    eigenframe_freq[idx].x = A0*product.x + B1*eigenframe_freq[idx].x;
    eigenframe_freq[idx].y = A0*product.y + B1*eigenframe_freq[idx].y;
  }
}

extern "C" __host__
void shift_image_add_to_eigenframe(CUDAstate_struct *restrict CUDAstate) {
  const int Nh  = CUDAstate->N_horizontal;
  const int Nv  = CUDAstate->N_vertical;
  const REAL A0 = CUDAstate->A0;
  const REAL B1 = CUDAstate->B1;

  // Reverse shifts are stored in aux_array3
  shift_image_add_to_eigenframe_GPU<<<MIN(Nv,512),MIN(Nh,512)>>>(Nh,Nv,A0,B1,
                                                                 CUDAstate->new_image_freq_domain,
                                                                 CUDAstate->aux_array3,
                                                                 CUDAstate->eigenframe_freq_domain);
}