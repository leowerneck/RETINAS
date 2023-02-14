#include "image_analysis.h"

/*
 *  Function: displacements_sub_pixel_estimate
 *  Author  : Leo Werneck
 *
 *  Compute the cross-correlation between the new and reference images.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see image_analysis.h).
 *
 *    displacements : out
 *      Horizontal and vertical displacements estimated to less than one pixel.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void displacements_sub_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements ) {

  // Step 1: Set auxiliary variables
  const REAL upsample_factor = state->upsample_factor;
  const REAL up_region_size  = CEIL(upsample_factor*1.5);
  const REAL dftshift        = round_towards_zero(up_region_size/2.0);
  const int S                = (int)up_region_size;
  const int SS               = S*S;

  // Step 2: Get maximum of upsampled image
  const int idx   = CBLAS_IAMAX_COMPLEX(SS, state->aux_array1, 1);
  const int i_max = idx/S;
  const int j_max = idx - i_max*S;

  // Step 3: Update the displacements
  displacements[0] += ((REAL)i_max - dftshift)/upsample_factor;
  displacements[1] += ((REAL)j_max - dftshift)/upsample_factor;
}
