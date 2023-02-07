#include "image_analysis.h"

#ifdef __NVCC__
#  define HOST_FUNC __host__
#else
#  define HOST_FUNC
#endif

#ifdef __cplusplus
#  define CFUNC extern "C"
#else
#  define CFUNC
#endif

#define CHOSTFUNC CFUNC HOST_FUNC

CHOSTFUNC
int max_real_2d(
    state_struct *restrict state,
    const int n,
    REAL *restrict x,
    int *restrict i_max,
    int *restrict j_max ) {
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
#ifdef __NVCC__
  cublasHandle_t h = state->cublasHandle;
  CUBLASIAMAX(h, n, x, 1, &i_max);
#else

#endif

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