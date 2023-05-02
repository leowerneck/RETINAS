#include "retinas.h"

/*
 *  Function: compute_displacements_and_add_new_image_to_sum
 *  Author  : Leo Werneck
 *
 *  Obtain the displacement between the new and reference images.
 *  Add the new image to the accumulator.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *    displacements : out
 *      Stores the horizontal and vertical displacements.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void compute_displacements_and_add_new_image_to_sum(
    state_struct *restrict state,
    REAL *restrict displacements ) {

  // Step 1: Compute the displacements via cross-correlation
  cross_correlate_ref_and_new_images(state);

  // Step 2: Get the full pixel estimate of the displacements
  displacements_full_pixel_estimate(state, displacements);

  // Step 3: Compute the displacements using upsampling
  if( (int)(state->upsample_factor+0.5) > 1 ) {
    upsample_around_displacements(state, displacements);
    displacements_sub_pixel_estimate(state, displacements);
  }

  // Step 4: Add new image to the sum
  add_new_image_to_sum(state);
}
