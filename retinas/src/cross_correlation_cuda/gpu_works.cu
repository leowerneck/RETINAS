#include "retinas.h"

__global__ void test_gpu() {
  printf("success!\n");
}

extern "C" __host__
bool gpu_works() {
  info("Testing if GPU works... ");
  fflush(stdout);
  test_gpu<<<1, 1>>>();
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("failed!\n");
    printf("Error name: %s", cudaGetErrorName(err));
    printf("Error msg : %s", cudaGetErrorString(err));
    return false;
  }

  return true;
}
