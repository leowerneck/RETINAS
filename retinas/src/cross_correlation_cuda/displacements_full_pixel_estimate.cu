#include "image_analysis.h"

extern "C" __host__
void displacements_full_pixel_estimate(
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
   *      displacements : Estimated to one pixel.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Set auxiliary variables
  const int Nh   = state->N_horizontal;
  const int Nv   = state->N_vertical;
  const int NhNv = Nh*Nv;

  // Step 2: Compute the absolute value of the cross-correlation
  absolute_value_2d(Nh, Nv, state->cross_correlation, state->aux_array_real);

  // Step 3: Compute index of the maximum
  const int idx_max = find_maxima(state->cublasHandle, NhNv, state->aux_array_real);
  // Flattened indices are computed using idx = i + Nh*j, so that
  // j = idx/Nh - i/Nh = idx/Nh,
  // because integer division i/Nh = 0 since i < Nh.
  // Once j is known, we can compute i using
  // i = idx - Nh*j
  const int j_max = idx_max/Nh;
  const int i_max = idx_max - j_max*Nh;
  displacements[0] = (REAL)i_max;
  displacements[1] = (REAL)j_max;

  // Step 4: If the displacements are larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacements[0] > Nh/2 ) displacements[0] -= Nh;
  if( displacements[1] > Nv/2 ) displacements[1] -= Nv;
}