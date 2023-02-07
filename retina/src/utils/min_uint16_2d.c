#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void min_uint16_2d(
    const int m,
    const int n,
    const uint16_t *restrict x,
    int *restrict i_min,
    int *restrict j_min ) {
  /*
   *  Finds the two-dimensional index of the minimum value of an array.
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
   *      i_min : Stores horizontal index so that x(i_min,j_min) is minimum.
   *      j_min : Stores vertical index so that x(i_min,j_min) is minimum.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Find the minimum index of the flattened array
  uint16_t min = x[0];
  int k_min=0;
  for(int k=1;k<m*n;k++) {
    if( x[k] < min ) {
      min   = x[k];
      k_min = k;
    }
  }

  // Step 2: For the two-dimensional indices, assume: k_min = i_min + m*j_min
  *j_min = k_min/m;
  *i_min = k_min - (*j_min)*m;
}
