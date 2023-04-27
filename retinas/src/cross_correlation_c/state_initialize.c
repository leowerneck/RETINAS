#include "retinas.h"

/*
 *  Function: state_initialize
 *  Author  : Leo Werneck
 *
 *  Initialize a new state object (see retinas.h).
 *
 *  Arguments
 *  ---------
 *    N_horizontal : in
 *      Number of horizontal points in the images.

 *    N_vertical : in
 *      Number of vertical points in the images.

 *    upsample_factor : in
 *      Upsampling factor.

 *    A0 : in
 *      See description for B1 below.

 *    B1 : in
 *      The reference image is updated according to
 *       ref_image = A0*new_image + B1*ref_image.
 *
 *    shot_noise_method : in
 *      Whether or not to use the shot noise method.
 *
 *    shift : in
 *      Shift to apply to the images before taking their reciprocal (only used
 *      if shot noise method is enabled).
 *
 *  Returns
 *  -------
 *    state : The state object, fully initialized.
 */
state_struct *state_initialize(
      const int N_horizontal,
      const int N_vertical,
      const REAL upsample_factor,
      const REAL A0,
      const REAL B1,
      const bool shot_noise_method,
      const REAL shift ) {

  info("Initializing state object.\n");
  info("  Parameters:\n");
  info("    N_horizontal    = %d\n", N_horizontal);
  info("    N_vertical      = %d\n", N_vertical);
  info("    upsample_factor = %g\n", upsample_factor);
  info("    A0              = %g\n", A0);
  info("    B1              = %g\n", B1);
  if( shot_noise_method )
    info("    shift           = %g\n", shift);

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
  const int aux_size     = MAX(N_horizontal,N_upsampling)*MAX(N_vertical,N_upsampling);

  // Step 4: Allocate memory for the auxiliary arrays
  state->aux_array1 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);
  state->aux_array2 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);
  state->aux_array3 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);

  // Step 5: Allocate memory for the arrays that hold the images
  state->new_image_time = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);
  state->new_image_freq = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);
  state->ref_image_freq = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);

  // Step 6: Create the FFT plans
  // Step 6.a: Forward FFT (the pointers here are dummy, they just need enough memory allocated)
  state->fft2_plan = FFTW_PLAN_DFT_2D(N_vertical, N_horizontal,
                                      state->new_image_time, state->new_image_freq,
                                      FFTW_FORWARD, FFTW_ESTIMATE);

  // Step 6.b: Inverse FFT (the pointers here are dummy, they just need enough memory allocated)
  state->ifft2_plan = FFTW_PLAN_DFT_2D(N_vertical, N_horizontal,
                                       state->new_image_time, state->new_image_freq,
                                       FFTW_BACKWARD, FFTW_ESTIMATE);

  // Step 7: Print basic information to the user
  info("Successfully initialized state object\n");

  // Step 8: Return C state
  return state;
}
