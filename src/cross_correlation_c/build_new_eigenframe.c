#include "image_analysis.h"

void build_next_eigenframe(
    const REAL *restrict displacements,
    Cstate_struct *restrict Cstate ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    displacements : Array containing the horizontal and vertical displacements.
   *    Cstate        : The C state object, containing the new eigenframe.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the reverse shift matrix
  compute_reverse_shift_matrix(Cstate->N_horizontal,
                               Cstate->N_vertical,
                               displacements,
                               Cstate->aux_array1,
                               Cstate->aux_array2,
                               Cstate->aux_array3);

  // Step 2: Now shift the new image and add it to the eigenframe
  for(int j=0;j<Cstate->N_vertical;j++) {
    for(int i=0;i<Cstate->N_horizontal;i++) {
      const int idx = i+Cstate->N_horizontal*j;
      Cstate->eigenframe_freq_domain[idx] =
        Cstate->A0*Cstate->new_image_freq_domain[idx]*Cstate->aux_array3[idx]
      + Cstate->B1*Cstate->eigenframe_freq_domain[idx];
    }
  }
}
