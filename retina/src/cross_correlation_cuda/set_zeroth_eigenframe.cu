#include "image_analysis.h"

__host__
void set_zeroth_eigenframe( state_struct *restrict state ) {
  /*
   *  Initialize zeroth eigenframe to the FFT of the new image.
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
  //         store it as the zeroth eigenframe.
  if( state->shot_noise_method )
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->reciprocal_new_image_time,
                    state->eigenframe_freq,
                    CUFFT_FORWARD);
  else
    FFT_EXECUTE_DFT(state->fft2_plan,
                    state->new_image_time,
                    state->eigenframe_freq,
                    CUFFT_FORWARD);

  // printf("Inside %s\n", __func__);
  // const int Nh = state->N_horizontal;
  // const int Nv = state->N_vertical;
  // printf("1/E = 1/(I+offset):\n");
  // print_2d_array_complex(Nh, Nv, state->new_image_time);
  // printf("F[1/E]:\n");
  // print_2d_array_complex(Nh, Nv, state->eigenframe_freq);
  // printf("Leaving %s\n", __func__);
}