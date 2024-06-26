#include "retinas.h"

extern "C" __host__
void compute_displacements_and_add_new_image_to_sum_shot_noise(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Obtain the displacement between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state         : The state object (see retinas.h).
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the displacements via cross-correlation
  cross_correlate_ref_and_new_images(state);

  // Step 2: Get the full pixel estimate of the displacements
  displacements_full_pixel_estimate_shot_noise(state, displacements);

  // Step 3: Compute the displacements using upsampling
  if( (int)(state->upsample_factor+0.5) > 1 ) {
    upsample_around_displacements(state, displacements);
    displacements_sub_pixel_estimate_shot_noise(state, displacements);
  }

  // Step 4: Before updating the refence image we must compute
  //         the FFT of the reciprocal of the new image
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->reciprocal_new_image_time,
                  state->new_image_freq,
                  CUFFT_FORWARD);

  // Step 5: Update the reference image
  add_new_image_to_sum(displacements, state);
}
