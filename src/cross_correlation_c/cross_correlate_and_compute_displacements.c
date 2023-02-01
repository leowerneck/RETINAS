#include "image_analysis.h"

void cross_correlate_and_compute_displacements(
    Cstate_struct *restrict Cstate,
    REAL *restrict displacements ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    Cstate        : The C state object, containing the new image.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the FFT of the new image
  FFTW_EXECUTE_DFT(Cstate->fft2_plan,Cstate->new_image_time_domain,Cstate->new_image_freq_domain);

  // Step 2: Compute image product target * src^{*}
  const int N_horizontal = Cstate->N_horizontal;
  const int N_vertical   = Cstate->N_vertical;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      Cstate->aux_array1[idx] = Cstate->new_image_freq_domain[idx]*CONJ(Cstate->eigenframe_freq_domain[idx]);
    }
  }

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFTW_EXECUTE_DFT(Cstate->ifft2_plan,Cstate->aux_array1,Cstate->aux_array2);

  // Step 4: Full-pixel estimate of the cross correlation maxima
  int i_max=0,j_max=0;
  REAL cc_max = -1.0;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      const REAL cc = CABS(Cstate->aux_array2[idx]);
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
}
