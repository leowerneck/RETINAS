#include "retinas.h"

extern "C" __host__
int find_maxima(
    cublasHandle_t h,
    const int n,
    REAL *restrict z) {
  /*
   *  Finds the index of the maximum element in an array. This function
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
  CUBLASIAMAX(h, n, z, 1, &i_max);
  return i_max-1;
}

extern "C" __host__
int find_minima(
    cublasHandle_t h,
    const int n,
    REAL *restrict z) {
  /*
   *  Finds the index of the minimum element in an array. This function
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
   *    i_min : The index corresponding to the minimum element in x.
   */

  int i_min;
  CUBLASIAMIN(h, n, z, 1, &i_min);
  return i_min-1;
}