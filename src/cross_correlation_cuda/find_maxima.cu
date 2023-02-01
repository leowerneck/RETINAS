#include "image_analysis.h"

// Find index of maximum element of an array (uses cuBLAS)
extern "C" __host__
int find_maxima(cublasHandle_t cublasHandle, const int n, REAL *restrict z) {
  int i_max;
  CUBLASIAMAX(cublasHandle,n,z,1,&i_max);
  return i_max-1;
}