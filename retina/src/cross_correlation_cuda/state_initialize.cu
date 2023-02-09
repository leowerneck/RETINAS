#include "image_analysis.h"

extern "C" __host__
state_struct *state_initialize(
    const int N_horizontal,
    const int N_vertical,
    const REAL upsample_factor,
    const REAL A0,
    const REAL B1,
    const REAL shift ) {
  /*
   *  Create a new CUDA state object.
   *
   *  Inputs
   *  ------
   *    N_horizontal    : Number of horizontal points in the images.
   *    N_vertical      : Number of vertical points in the images.
   *    upsample_factor : Upsampling factor.
   *    A0              : See description for B1 below.
   *    B1              : The new eigenframe is computed according to
   *                      eigenframe = A0*new_image + B1*eigenframe.
   *    shift           : In the shot-noise algorithm, this is added to
   *                      each new image before taking its reciprocal.
   *
   *  Returns
   *  -------
   *    state           : The state object, fully initialized.
   */

  info("Initializing state object\n");
  info("    Input parameters:\n");
  info("        N_horizontal    = %d\n", N_horizontal);
  info("        N_vertical      = %d\n", N_vertical);
  info("        upsample_factor = %g\n", upsample_factor);
  info("        A0              = %g\n", A0);
  info("        B1              = %g\n", B1);
  if(shift >= 0)
    info("        shift           = %g\n", shift);

  // Step 1: Allocate memory for the parameter struct
  state_struct *state = (state_struct *)malloc(sizeof(state_struct));

  // Step 2: Copy Python parameters to the C state struct
  state->N_horizontal    = N_horizontal;
  state->N_vertical      = N_vertical;
  state->upsample_factor = upsample_factor;
  state->A0              = A0;
  state->B1              = B1;
  state->shift           = shift;

  // Step 3: Define auxiliary variables
  const int NhNv         = N_horizontal * N_vertical;
  const int N_upsampling = (int)CEIL(upsample_factor * 1.5);
  const int aux_size     = MAX(N_horizontal, N_upsampling)*MAX(N_vertical, N_upsampling);

  // Step 4: Allocate memory for host quantities
  state->host_aux_array = (COMPLEX *)malloc(sizeof(COMPLEX)*NhNv);

  // Step 5: Allocate memory for device quantities
  // Step 5.a: Auxiliary arrays
  cudaMalloc(&state->aux_array_int , sizeof(uint16_t)*aux_size);
  cudaMalloc(&state->aux_array_real, sizeof(REAL)    *aux_size);
  cudaMalloc(&state->aux_array1    , sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array2    , sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array3    , sizeof(COMPLEX) *aux_size);

  // Step 5.b: Arrays that hold the images
  cudaMalloc(&state->new_image_time , sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->new_image_freq , sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->reciprocal_new_image_time, sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->eigenframe_freq, sizeof(COMPLEX)*NhNv);

  // Step 5.c: Create the FFT plan
  cufftPlan2d(&state->fft2_plan, N_vertical, N_horizontal, FFT_C2C);

  // Step 5.d: Create the cuBLAS handle
  cublasCreate(&state->cublasHandle);

  // Step 6: Print basic information to the user
  info("Successfully initialized state object\n");

  // Step 7: All done! Return the CUDA state object
  return state;
}