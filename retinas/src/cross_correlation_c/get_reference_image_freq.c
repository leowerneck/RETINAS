#include "retinas.h"

/*
 *  Function: get_reference_image_freq
 *  Author  : Leo Werneck
 *
 *  Returns the FFT of the current reference image.
 *
 *  Arguments
 *  ---------
 *    state : in
 *      The state object (see retinas.h).
 *
 *    ref_image_freq : out
 *      Complex array that stores the FFT of the reference image.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void get_reference_image_freq(
    state_struct *restrict state,
    COMPLEX *restrict ref_image_freq ) {

  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i + state->N_horizontal * j;
      ref_image_freq[idx] = state->ref_image_freq[idx];
    }
  }
}
