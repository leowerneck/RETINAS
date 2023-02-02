#include "image_analysis.h"

void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    displacements : Array containing the horizontal and vertical displacements.
   *    state        : The C state object, containing the new eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

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
      state->eigenframe_freq_domain[idx] =
        state->A0*state->new_image_freq_domain[idx]*state->aux_array3[idx]
      + state->B1*state->eigenframe_freq_domain[idx];
    }
  }
}
