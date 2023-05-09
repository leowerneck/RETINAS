#include "retinas.h"

/*
 *  Function: add_new_image_to_sum
 *  Author  : Leo Werneck
 *
 *  Add the new image to the accumulator.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void add_new_image_to_sum(
    const REAL *restrict displacements,
    state_struct *restrict state ) {

  // Step 1: Compute the reverse shift matrix
  compute_reverse_shift_matrix(state->N_horizontal,
                               state->N_vertical,
                               displacements,
                               state->aux_array1,
                               state->aux_array2,
                               state->aux_array3);

  // Step 2: Now shift the new image and add it to the accumulator
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i+state->N_horizontal*j;
      state->image_sum_freq[idx] += state->new_image_freq[idx]*state->aux_array3[idx];
    }
  }
  state->image_counter++;
}
