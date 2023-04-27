#include "image_analysis.h"

extern "C" __host__
void cross_correlate_ref_and_new_images( state_struct *restrict state ) {
  /*
   *  Compute the cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state : The state object, containing the new image.
   *
   *  Outputs
   *  -------
   *    state : The state object, containing the cross correlation.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the FFT of the new image
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->new_image_time,
                  state->new_image_freq,
                  CUFFT_FORWARD);



  // Step 2: Compute image product target * src^{*}
  element_wise_multiplication_conj_2d(state->N_horizontal,
                                      state->N_vertical,
                                      state->new_image_freq,
                                      state->ref_image_freq,
                                      state->image_product);

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->image_product,
                  state->cross_correlation,
                  CUFFT_INVERSE);
}
