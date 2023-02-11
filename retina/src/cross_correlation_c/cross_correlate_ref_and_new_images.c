#include "image_analysis.h"

void cross_correlate_ref_and_new_images( state_struct *restrict state ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state        : The C state object, containing the new image.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the FFT of the new image
  FFTW_EXECUTE_DFT(state->fft2_plan,state->new_image_time,state->new_image_freq);

  // Step 2: Compute image product target * src^{*}
  const int N_horizontal = state->N_horizontal;
  const int N_vertical   = state->N_vertical;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      state->aux_array1[idx] = state->new_image_freq[idx]*CONJ(state->ref_image_freq[idx]);
    }
  }

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFTW_EXECUTE_DFT(state->ifft2_plan,state->aux_array1,state->aux_array2);
}
