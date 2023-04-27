#include "retinas.h"

/*
 *  Function: set_first_reference_image
 *  Author  : Leo Werneck
 *
 *  Initialize initial reference image to the FFT of the new image.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *     Nothing.
 */
void set_first_reference_image( state_struct *restrict state ) {


  // Step 1: Compute FFT of the new_image_time and
  //         store it as the initial reference image.
  FFTW_EXECUTE_DFT(state->fft2_plan,
                   state->new_image_time,
                   state->ref_image_freq);
}
