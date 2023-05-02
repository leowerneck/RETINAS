#include "image_analysis.h"

extern "C" __host__
state_struct *state_initialize(
    const int N_horizontal,
    const int N_vertical,
    const REAL upsample_factor,
    const REAL A0,
    const REAL B1,
    const bool shot_noise_method,
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
   *    B1              : The reference image is updated according to
   *                      ref_image = A0*new_image + B1*ref_imagex.
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
  if(shot_noise_method)
    info("        shift           = %g\n", shift);

  // Step 1: Allocate memory for the parameter struct
  state_struct *state = (state_struct *)malloc(sizeof(state_struct));

  // Step 2: Copy Python parameters to the C state struct
  state->N_horizontal      = N_horizontal;
  state->N_vertical        = N_vertical;
  state->upsample_factor   = upsample_factor;
  state->A0                = A0;
  state->B1                = B1;
  state->shot_noise_method = shot_noise_method;
  state->shift             = shift;

  // Step 3: Define auxiliary variables
  const int NhNv         = N_horizontal * N_vertical;
  const int N_upsampling = (int)CEIL(upsample_factor * 1.5);
  const int aux_size     = MAX(N_horizontal, N_upsampling)*MAX(N_vertical, N_upsampling);

  // Step 4: Allocate memory for device quantities
  // Step 4.a: Auxiliary arrays
  cudaMalloc(&state->aux_array_int , sizeof(uint16_t)*aux_size);
  cudaMalloc(&state->aux_array_real, sizeof(REAL)    *aux_size);
  cudaMalloc(&state->aux_array1    , sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array2    , sizeof(COMPLEX) *aux_size);
  cudaMalloc(&state->aux_array3    , sizeof(COMPLEX) *aux_size);

  // Step 4.b: Arrays that hold the images
  cudaMalloc(&state->new_image_time           , sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->new_image_freq           , sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->reciprocal_new_image_time, sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->ref_image_freq           , sizeof(COMPLEX)*NhNv);
  cudaMalloc(&state->image_sum_freq           , sizeof(COMPLEX)*NhNv);

  // Step 4.c: Set additional, auxiliary arrays
  // Step 4.c.1: Pointers that map to aux_array1
  state->ref_image_time    = state->aux_array1;
  state->image_product     = state->aux_array1;
  state->upsampled_image   = state->aux_array1;
  state->horizontal_shifts = state->aux_array1;
  // Step 4.c.2: Pointers that map to aux_array2
  state->cross_correlation = state->aux_array2;
  state->horizontal_kernel = state->aux_array2;
  state->vertical_kernel   = state->aux_array2;
  state->vertical_shifts   = state->aux_array2;
  // Step 4.c.3: Pointers that map to aux_array3
  state->partial_product   = state->aux_array3;
  state->shift_matrix      = state->aux_array3;

  // Step 4.d: Create the FFT plan
  cufftPlan2d(&state->fft2_plan, N_vertical, N_horizontal, FFT_C2C);

  // Step 4.e: Create the cuBLAS handle
  cublasCreate(&state->cublasHandle);

  // Step 5: Print basic information to the user
  info("Successfully initialized state object\n");

  // Step 6: All done! Return the state object
  return state;
}