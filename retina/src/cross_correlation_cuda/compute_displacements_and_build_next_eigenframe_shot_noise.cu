#include "image_analysis.h"

extern "C" __host__
void compute_displacements_and_build_next_eigenframe_shot_noise(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Obtain the displacement between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state         : The CUDA state object, containing the new and reference images.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the displacements via cross-correlation
  cross_correlate_ref_and_new_images(state);

  // const int Nh = state->N_horizontal;
  // const int Nv = state->N_vertical;
  // printf("I^2:\n");
  // print_2d_array_complex(Nh, Nv, state->new_image_time);
  // printf("F[I^2]:\n");
  // print_2d_array_complex(Nh, Nv, state->new_image_freq);
  // printf("F[1/E]:\n");
  // print_2d_array_complex(Nh, Nv, state->eigenframe_freq);
  // printf("Image product (aux_array1):\n");
  // print_2d_array_complex(Nh, Nv, state->aux_array1);
  // printf("Cross-correlation (aux_array2):\n");
  // print_2d_array_complex(Nh, Nv, state->aux_array2);

  // Step 2: Get the full pixel estimate of the displacements
  displacements_full_pixel_estimate_shot_noise(state, displacements);

  // printf("Full pixel displacements: %.15e %.15e\n", displacements[0], displacements[1]);

  // Step 3: Compute the displacements using upsampling
  if( (int)(state->upsample_factor+0.5) > 1 ) {
    upsample_around_displacements(state, displacements);
    displacements_sub_pixel_estimate_shot_noise(state, displacements);
  }

  // Step 4: Before building the next eigenframe we must compute
  //         the FFT of the reciprocal of the new image
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->reciprocal_new_image_time,
                  state->new_image_freq,
                  CUFFT_FORWARD);

  // const int Nh = state->N_horizontal;
  // const int Nv = state->N_vertical;
  // printf("(1/I+s):\n");
  // print_2d_array_complex(Nh, Nv, state->reciprocal_new_image_time);
  // printf("F[(1/I+s)]:\n");
  // print_2d_array_complex(Nh, Nv, state->new_image_freq);

  // Step 5: Build next eigenframe
  build_next_eigenframe(displacements, state);
}
