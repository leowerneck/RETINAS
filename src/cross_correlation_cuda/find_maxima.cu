#include "image_analysis.h"

// Find index of maximum element of an array (uses cuBLAS)
extern "C" __host__
int find_maxima(
    cublasHandle_t h,
    const int n,
    REAL *restrict x ) {
  /*
   *  Performs the matrix multiplication C = A.B. This function
   *  uses cuBLAS.
   *
   *  Inputs
   *  ------
   *    h : cuBLAS handle object.
   *    n : Size of the array.
   *    x : Real array of size x.
   *
   *  Returns
   *  -------
   *    i_max : The index corresponding to the maximum element in x.
   */

  int i_max;
  CUBLASIAMAX(h, n, x, 1, &i_max);
  return i_max-1;
}