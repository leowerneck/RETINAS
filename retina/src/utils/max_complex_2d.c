#include "image_analysis.h"

void max_complex_2d(
    const int m,
    const int n,
    COMPLEX *restrict x,
    int *restrict i_max,
    int *restrict j_max ) {
  /*
   *  Finds the index of the maximum element in an array. This function
   *  uses CBLAS.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      m : Horizontal size of the array.
   *      n : Vertical size of the array.
   *      x : Complex array of size m*n.
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

  // Step 1: Find the maximum of the array
  int k_max = CBLAS_IAMAX_COMPLEX(n, x, 1);

  // Step 2: For the two-dimensional indices, assume: k_max = i_max + m*j_max
  *j_max = k_max/m;
  *i_max = k_max - (*j_max)*m;
}
