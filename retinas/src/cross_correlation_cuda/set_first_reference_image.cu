#include "retinas.h"

__global__
static void init_image_sum_to_ref_image_gpu(
    const int n,
    COMPLEX *restrict ref_image_freq,
    COMPLEX *restrict image_sum_freq ) {

  const int index  = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for(int i=index;i<n;i+=stride)
    image_sum_freq[i] = ref_image_freq[i];
}

__host__
void set_first_reference_image( state_struct *restrict state ) {
  /*
   *  Initialize first reference image to the FFT of the new image.
   *
   *  Inputs
   *  ------
   *    state : The C state object, containing the new image.
   *
   *  Returns
   *  -------
   *     Nothing.
   */

  // Step 1: Compute FFT of the new_image_time and
  //         store it as the first reference image.
  if( state->shot_noise_method )
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->reciprocal_new_image_time,
                    state->ref_image_freq,
                    CUFFT_FORWARD);
  else
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->new_image_time,
                    state->ref_image_freq,
                    CUFFT_FORWARD);

  const int Nh   = state->N_horizontal;
  const int Nv   = state->N_vertical;
  const int NhNv = Nh*Nv;
  init_image_sum_to_ref_image_gpu<<<MIN(Nv,512),MIN(Nh,512)>>>(
      NhNv, state->ref_image_freq, state->image_sum_freq);

  state->image_counter = 1;
}