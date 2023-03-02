#include "retina.h"

/*
 *  Function: cross_correlate_ref_and_new_images
 *  Author  : Leo Werneck
 *
 *  Find the displacements by finding the maxima of the
 *  cross-correlation between the new and reference images.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retina.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void cross_correlate_ref_and_new_images( state_struct *restrict state ) {

  // Step 1: Compute the FFT of the new image
  FFTW_EXECUTE_DFT(state->fft2_plan, state->new_image_time, state->new_image_freq);

  // Step 2: Compute image product target * src^{*}
  const int N_total = state->N_horizontal*state->N_vertical;
  for(int i=0;i<N_total;i++)
    state->aux_array1[i] = state->new_image_freq[i]*CONJ(state->ref_image_freq[i]);

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFTW_EXECUTE_DFT(state->ifft2_plan, state->aux_array1, state->aux_array2);
}
