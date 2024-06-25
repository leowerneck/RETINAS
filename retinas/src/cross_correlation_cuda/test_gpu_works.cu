void test_if_gpu_works_() {
  printf("success!\n");
}

bool test_if_gpu_works() {
  printf("Testing if GPU works... ");
  fflush(stdout);
  test_if_gpu_works_<<<1, 1>>>();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("failed!\n");
    printf("Error name: %s", cudaGetErrorName(err));
    printf("Error msg : %s", cudaGetErrorString(err));
    return false;
  }

  return true;
}
