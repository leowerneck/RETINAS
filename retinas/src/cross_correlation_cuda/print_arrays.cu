#include "retinas.h"

__global__
void print_1d_array_complex_gpu(
    const int n,
    const COMPLEX *restrict z ) {
  for(int i=0;i<n;i++)
    printf(" (%23.15e,%23.15e)", z[i].x, z[i].y);
  printf("\n");
}

__global__
void print_2d_array_real_gpu(
    const int m,
    const int n,
    const REAL *restrict x ) {

  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      printf("%23.15e ", x[i+m*j]);
    }
    printf("\n");
  }
}

__global__
void print_2d_array_complex_gpu(
    const int m,
    const int n,
    const COMPLEX *restrict z ) {

  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      printf("%23.15e %23.15e, ", z[i+m*j].x, z[i+m*j].y);
    }
    printf("\n");
  }
}

__host__
void print_1d_array_complex(
    const int n,
    const COMPLEX *restrict z ) {

  print_1d_array_complex_gpu<<<1,1>>>(n, z);
  cudaDeviceSynchronize();
}

__host__
void print_2d_array_real(
    const int m,
    const int n,
    const REAL *restrict x ) {

  print_2d_array_real_gpu<<<1,1>>>(m, n, x);
  cudaDeviceSynchronize();
}

__host__
void print_2d_array_complex(
    const int m,
    const int n,
    const COMPLEX *restrict z ) {

  print_2d_array_complex_gpu<<<1,1>>>(m, n, z);
  cudaDeviceSynchronize();
}