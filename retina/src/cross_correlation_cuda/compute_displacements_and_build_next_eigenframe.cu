#include "image_analysis.h"

extern "C" __host__
void compute_displacements_and_build_next_eigenframe(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Obtain the displacement between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state         : The CUDA state object, containing the new and reference images.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the displacements via cross-correlation
  cross_correlate_ref_and_new_images(state);

  // Step 2: Get the full pixel estimate of the displacements
  displacements_full_pixel_estimate(state, displacements);

  // Step 2: Compute the displacements using upsampling
  if( (int)(state->upsample_factor+0.5) > 1 ) {
    upsample_around_displacements(state, displacements);
    displacements_sub_pixel_estimate(state, displacements);
  }

  // Step 3: Build next eigenframe
  build_next_eigenframe(displacements, state);
}