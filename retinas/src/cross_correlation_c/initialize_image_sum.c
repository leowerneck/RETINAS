#include "retinas.h"

/*
 *  Function: initialize_image_sum
 *  Author  : Leo Werneck
 *
 *  Initialize the image sum to the reference image.
 *
 *  Arguments
 *  ---------
 *    state : in/out
 *      The state object (see retinas.h).
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void initialize_image_sum( state_struct *restrict state ) {

  for(int j=0;j<state->N_vertical;j++) {
    for(int i=0;i<state->N_horizontal;i++) {
      const int idx = i+state->N_horizontal*j;
      state->image_sum_freq[idx] = state->ref_image_freq[idx];
    }
  }
  state->image_counter = 1;
}
