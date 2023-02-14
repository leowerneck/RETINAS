#include "image_analysis.h"

/*
 *  Function: set_zeroth_eigenframe
 *  Author  : Leo Werneck
 *
 *  Initialize zeroth eigenframe to the FFT of the new image.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see image_analysis.h).
 *
 *  Returns
 *  -------
 *     Nothing.
 */
void set_zeroth_eigenframe( state_struct *restrict state ) {


  // Step 1: Compute FFT of the new_image_time and
  //         store it as the zeroth eigenframe.
  FFTW_EXECUTE_DFT(state->fft2_plan,
                   state->new_image_time,
                   state->ref_image_freq);
}
