#include "image_analysis.h"

// Find index of maximum element of an array (uses cuBLAS)
extern "C" __host__
int find_minima(cublasHandle_t cublasHandle, const int n, REAL *restrict z) {
  int i_min;
  CUBLASIAMIN(cublasHandle,n,z,1,&i_min);
  return i_min-1;
}