#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void max_uint16_2d(
    const int m,
    const int n,
    const uint16_t *restrict x,
    int *restrict i_max,
    int *restrict j_max ) {
  /*
   *  Finds the two-dimensional index of the maximum value of an array.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      m : Horizontal size of the array.
   *      n : Vertical size of the array.
   *      x : Unsigned integer array of size m*n.
   *
   *    Outputs
   *    -------
   *      i_max : Stores horizontal index so that x(i_max,j_max) is maximum.
   *      j_max : Stores vertical index so that x(i_max,j_max) is maximum.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Find the maximum index of the flattened array
  uint16_t max = x[0];
  int k_max=0;
  for(int k=1;k<m*n;k++) {
    if( x[k] > max ) {
      max   = x[k];
      k_max = k;
    }
  }

  // Step 2: For the two-dimensional indices, assume: k_max = i_max + m*j_max
  *j_max = k_max/m;
  *i_max = k_max - (*j_max)*m;
}
