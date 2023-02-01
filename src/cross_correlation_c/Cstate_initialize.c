#include "image_analysis.h"

Cstate_struct *Cstate_initialize(
      const int N_horizontal,
      const int N_vertical,
      const REAL upsample_factor,
      const REAL A0,
      const REAL B1 ) {
  /*
   *  Create a new C state object.
   *
   *  Inputs
   *  ------
   *    N_horizontal    : Number of horizontal points in the images.
   *    N_vertical      : Number of vertical points in the images.
   *    upsample_factor : Upsampling factor.
   *    A0              : See description for B1 below.
   *    B1              : The new eigenframe is computed according to
   *                      eigenframe = A0*new_image + B1*eigenframe.
   *
   *  Returns
   *  -------
   *    Cstate          : The C state object, fully initialized.
   */

  info("Initializing C state object.\n");
  info("  Parameters:\n");
  info("    N_horizontal    = %d\n", N_horizontal);
  info("    N_vertical      = %d\n", N_vertical);
  info("    upsample_factor = %g\n", upsample_factor);
  info("    A0              = %g\n", A0);
  info("    B1              = %g\n", B1);

  // Step 1: Allocate memory for the parameter struct
  Cstate_struct *Cstate = (Cstate_struct *)malloc(sizeof(Cstate_struct));

  // Step 2: Copy Python parameters to the C state struct
  Cstate->N_horizontal    = N_horizontal;
  Cstate->N_vertical      = N_vertical;
  Cstate->upsample_factor = upsample_factor;
  Cstate->A0              = A0;
  Cstate->B1              = B1;

  // Step 3: Define auxiliary variables
  const int NhNv         = N_horizontal * N_vertical;
  const int N_upsampling = (int)CEIL(upsample_factor * 1.5);
  const int aux_size     = MAX(N_horizontal,N_upsampling)*MAX(N_vertical,N_upsampling);

  // Step 4: Allocate memory for the auxiliary arrays
  Cstate->aux_array1 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);
  Cstate->aux_array2 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);
  Cstate->aux_array3 = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(aux_size);

  // Step 5: Allocate memory for the arrays that hold the images
  Cstate->new_image_time_domain  = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);
  Cstate->new_image_freq_domain  = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);
  Cstate->eigenframe_freq_domain = (COMPLEX *restrict)FFTW_ALLOC_COMPLEX(NhNv);

  // Step 6: Create the FFT plans
  // Step 6.a: Forward FFT (the pointers here are dummy, they just need enough memory allocated)
  Cstate->fft2_plan = FFTW_PLAN_DFT_2D(N_vertical, N_horizontal,
                                       Cstate->new_image_time_domain, Cstate->new_image_freq_domain,
                                       FFTW_FORWARD, FFTW_ESTIMATE);

  // Step 6.b: Inverse FFT (the pointers here are dummy, they just need enough memory allocated)
  Cstate->ifft2_plan = FFTW_PLAN_DFT_2D(N_vertical, N_horizontal,
                                        Cstate->new_image_time_domain, Cstate->new_image_freq_domain,
                                        FFTW_BACKWARD, FFTW_ESTIMATE);

  // Step 7: Print basic information to the user
  info("Successfully initialized C state object\n");

  // Step 8: Return C state
  return Cstate;
}
