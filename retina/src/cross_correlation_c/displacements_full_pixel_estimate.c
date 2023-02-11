#include "image_analysis.h"

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
  cosnt int NhNv = Nh*Nv;

  // Step 1: Full pixel estimate of the cross-correlation maxima
  int i_max=0,j_max=0;
  REAL cc_max = -1.0;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      const REAL cc = CABS(state->aux_array2[idx]);
      if( cc > cc_max ) {
        cc_max = cc;
        i_max  = i;
        j_max  = j;
      }
    }
  }
  displacements[0] = (REAL)i_max;
  displacements[1] = (REAL)j_max;

  // Step 5: Compute midpoint
  const int midpoint[2] = {N_horizontal/2,N_vertical/2};

  // Step 6: If the displacements are larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacements[0] > midpoint[0] ) displacements[0] -= N_horizontal;
  if( displacements[1] > midpoint[1] ) displacements[1] -= N_vertical;
