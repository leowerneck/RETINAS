#include "image_analysis.h"

int compute_displacements(
    Cstate_struct *restrict Cstate,
    REAL *restrict displacements ) {
  /*
   *  Obtain the displacement between the new and reference images.
   *
   *  Inputs
   *  ------
   *    Cstate : Cstate_struct *
   *      The C state object, containing the new and reference images.
   *    displacements : REAL *
   *      Stores the result.
   *
   *  Returns
   *  -------
   *    error_key : int
   *      0 - success
   */

  // Step 1: Compute the displacements via cross-correlation
  get_displacements_by_cross_correlation(Cstate, displacements);

  // Step 2: Compute the displacements using upsampling
  if( (int)(Cstate->upsample_factor+0.5) > 1 )
    get_subpixel_displacements_by_upsampling(Cstate, displacements);

  // Step 3: Build next eigenframe
  build_next_eigenframe(displacements, Cstate);

  // All done!
  return 0;
}
