#include "retinas.h"

/*
 *  Function: update_reference_image_from_image_sum
 *  Author  : Leo Werneck
 *
 *  Update the reference image based on the images in the accumulator.
 *
 *  Arguments
 *  ---------
 *    displacements : in
 *      Array containing the horizontal and vertical displacements.
 *
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void update_reference_image_from_image_sum( state_struct *restrict state ) {

  const REAL norm = 1.0/((REAL)state->image_counter);
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i+state->N_horizontal*j;
      state->ref_image_freq[idx] = state->image_sum_freq[idx]*norm;
      state->image_sum_freq[idx] = state->ref_image_freq[idx];
    }
  }
  state->image_counter = 1;
}
