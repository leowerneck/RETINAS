#include "retinas.h"

/*
 *  Compute the cross-correlation between the new and reference images.
 *
 *  Arguments
 *  ---------
 *    state : in
 *      The state object (see retinas.h).
 *
 *    displacements : out
 *      Horizontal and vertical displacements estimated to one pixel.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void displacements_full_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements ) {

  // Step 1: Set auxiliary variables
  const int Nh   = state->N_horizontal;
  const int Nv   = state->N_vertical;
  const int NhNv = Nh*Nv;

  // Step 2: Full pixel estimate of the cross-correlation maxima
  const int idx   = CBLAS_IAMAX_COMPLEX(NhNv, state->aux_array2, 1);
  const int j_max = idx/Nh;
  const int i_max = idx - j_max*Nh;
  displacements[0] = (REAL)i_max;
  displacements[1] = (REAL)j_max;

  // Step 3: If the displacements are larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacements[0] > Nh/2 ) displacements[0] -= Nh;
  if( displacements[1] > Nv/2 ) displacements[1] -= Nv;
}
