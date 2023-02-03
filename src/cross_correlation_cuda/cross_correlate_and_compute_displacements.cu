#include "image_analysis.h"

extern "C" __host__
void cross_correlate_and_compute_displacements(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Find the displacements by finding the maxima of the
   *  cross-correlation between the new and reference images.
   *
   *  Inputs
   *  ------
   *    state        : The C state object, containing the new image.
   *    displacements : Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  // Step 1: Compute the FFT of the new image
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->new_image_time_domain,
                  state->new_image_freq_domain,
                  CUFFT_FORWARD);

  // Step 2: Compute image product target * src^{*}
  const int Nh   = state->N_horizontal;
  const int Nv   = state->N_vertical;
  const int NhNv = Nh*Nv;
  element_wise_multiplication_conj_2d(Nh, Nv,
                                      state->new_image_freq_domain,
                                      state->eigenframe_freq_domain,
                                      state->aux_array1);

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFT_EXECUTE_DFT(state->fft2_plan,
                  state->aux_array1,
                  state->aux_array2,
                  CUFFT_INVERSE);

  // Step 4: Full-pixel estimate of the cross correlation maxima
  absolute_value_2d(Nh, Nv, state->aux_array2, state->aux_array_real);
  const int idx_max = find_maxima(state->cublasHandle,NhNv,state->aux_array_real);
  // Flattened indices are computed using idx = i + Nh*j, so that
  // j = idx/Nh - i/Nh = idx/Nh,
  // because integer division i/Nh = 0 since i < Nh.
  // Once j is known, we can compute i using
  // i = idx - Nh*j
  const int j_max = idx_max/Nh;
  const int i_max = idx_max - j_max*Nh;
  displacement[0] = (REAL)i_max;
  displacement[1] = (REAL)j_max;

  // Step 5: Compute midpoint
  const int midpoint[2] = {Nh/2,Nv/2};

  // Step 6: If the displacement is larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacement[0] > midpoint[0] ) displacement[0] -= Nh;
  if( displacement[1] > midpoint[1] ) displacement[1] -= Nv;
}
