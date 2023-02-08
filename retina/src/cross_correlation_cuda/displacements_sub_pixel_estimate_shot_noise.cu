#include "image_analysis.h"

extern "C" __host__
void displacements_sub_pixel_estimate_shot_noise(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Compute the cross-correlation between the new and reference images.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      state : The state object, containing the new image
   *
   *    Outputs
   *    -------
   *      displacements : Estimated to less than one pixel.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Set auxiliary variables
  const REAL upsample_factor = state->upsample_factor;
  const REAL up_region_size  = CEIL(upsample_factor*1.5);
  const REAL dftshift        = round_towards_zero(up_region_size/2.0);
  const int S                = (int)up_region_size;
  const int SS               = S*S;

  // Step 2: Compute the absolute value of the cross-correlation
  absolute_value_2d(S, S, state->aux_array1, state->aux_array_real);

  // Step 3: Compute index of the minimum
  const int idx_min = find_minima(state->cublasHandle, SS, state->aux_array_real);
  const int i_min   = idx_min/S;
  const int j_min   = idx_min - i_min*S;

  // Step 4: Update the displacements
  displacements[0] += ((REAL)i_min - dftshift)/upsample_factor;
  displacements[1] += ((REAL)j_min - dftshift)/upsample_factor;
}