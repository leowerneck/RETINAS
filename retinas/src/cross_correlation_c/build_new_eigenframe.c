#include "retinas.h"

/*
 *  Function: build_next_eigenframe
 *  Author  : Leo Werneck
 *
 *  Constructs the next reference image using:
 *   ref_image_new = A0*new_image_shifted + B1*ref_image_old.
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
void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state ) {

  // Step 1: Compute the reverse shift matrix
  compute_reverse_shift_matrix(state->N_horizontal,
                               state->N_vertical,
                               displacements,
                               state->aux_array1,
                               state->aux_array2,
                               state->aux_array3);

  // Step 2: Now shift the new image and add it to the eigenframe
  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i+state->N_horizontal*j;
      state->ref_image_freq[idx] =
        state->A0*state->new_image_freq[idx]*state->aux_array3[idx]
      + state->B1*state->ref_image_freq[idx];
    }
  }
}
