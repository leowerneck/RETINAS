#include "image_analysis.h"

REAL typecast_input_image_rebin_4x4_and_compute_brightness(
    const uint16_t *restrict input_array,
    Cstate_struct *restrict Cstate ) {
  /*
   *  Typecast the input image from uint16 to REAL. In the process,
   *  the image is rebinned to a imsage of size (Nh/4, Nv/4), with
   *  each pixel of the binned image set to the average of 16 pixels
   *  of the original image. Also compute the brightness, which is
   *  the sum of all pixel values in the image.
   *
   *  Inputs
   *  ------
   *    input_array : Input image stored as a 1D array.
   *    Cstate      : Pointer to the Cstate object.
   *
   *  Returns
   *  -------
   *     brightness : Brightness of the image.
   */

  // Step 1: Initialize local sum to zero
  REAL brightness = 0.0;

  // Step 2: Loop over the array, summing its entries
  //         typecasting it, and binning to the output array
  const int N_horizontal_original = 4*Cstate->N_horizontal;
  const int N_vertical_original   = 4*Cstate->N_vertical;
  for(int j=0;j<N_vertical_original;j+=4) {
    for(int i=0;i<N_horizontal_original;i+=4) {
      // Step 3.a: Sum all values in the current bin
      REAL bin_value = 0.0;
      for(int jj=0;jj<4;jj++) {
        for(int ii=0;ii<4;ii++) {
          // Step 3.a.i: This is where we typecast
          bin_value += (REAL)input_array[(i+ii) + N_horizontal_original*(j+jj)];
        }
      }
      // Step 3.b: Set the binned array to the pixel sum
      Cstate->new_image_time_domain[(i + Cstate->N_horizontal*j)/4] = bin_value;

      // Step 3.c: Add pixel sum to the total accumulated sum
      brightness += bin_value;
    }
  }

  // Step 4: Return the brightness
  return brightness;
}
