#include "image_analysis.h"

void compute_displacements_and_build_next_eigenframe(
    Cstate_struct *restrict Cstate,
    REAL *restrict displacements ) {
  /*
   *  Obtain the displacement between the new and reference images.
   *
   *  Inputs
   *  ------
   *    Cstate        : The C state object, containing the new and reference images.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the displacements via cross-correlation
  cross_correlate_and_compute_displacements(Cstate, displacements);

  // Step 2: Compute the displacements using upsampling
  if( (int)(Cstate->upsample_factor+0.5) > 1 )
    upsample_and_compute_subpixel_displacements(Cstate, displacements);

  // Step 3: Build next eigenframe
  build_next_eigenframe(displacements, Cstate);
}
