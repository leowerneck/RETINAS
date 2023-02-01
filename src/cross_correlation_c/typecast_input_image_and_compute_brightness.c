#include "image_analysis.h"

REAL typecast_input_image_and_compute_brightness(
    const uint16_t *restrict input_array,
    Cstate_struct *restrict Cstate ) {
  /*
   *  Typecast the input image from uint16 to REAL. Also compute the
   *  brightness, which is the sum of all pixel values in the image.
   *
   *  Inputs
   *  ------
   *    input_array : uint16_t *
   *      Input image stored as a 1D array.
   *    Cstate : Cstate_struct *
   *      Pointer to the Cstate object.
   *
   *  Returns
   *  -------
   *     brightness : REAL
   *       Brightness of the image.
   */

  // Step 1: Initialize local sum to zero
  REAL brightness = 0.0;

  // Step 2: Loop over the array, summing its entries
  //         and typecasting to the output array
  for(int j=0;j<Cstate->N_vertical;j++) {
    for(int i=0;i<Cstate->N_horizontal;i++) {
      // Step 2.a: Set local index
      const int idx = i + Cstate->N_horizontal*j;

      // Step 2.b: Typecast input array value
      const REAL value = (REAL)input_array[idx];

      // Step 2.c: Write to output array
      Cstate->new_image_time_domain[idx] = value + I*0.0f;

      // Step 2.d: Add to total sum
      brightness += value;
    }
  }

  // Step 3: Return the brightness
  return brightness;
}
