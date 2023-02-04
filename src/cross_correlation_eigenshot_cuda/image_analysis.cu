#include "image_analysis.h"

__host__
int set_zeroth_eigenframe( CUDAstate_struct *restrict CUDAstate ) {

  // Step 1: Compute FFT of the new_image_time and
  //         store it as the zeroth eigenframe.
  FFT_EXECUTE_DFT(CUDAstate->fft2_plan,
                  CUDAstate->reciprocal_new_image_time,
                  CUDAstate->reciprocal_eigenframe_freq,
                  CUFFT_FORWARD);

  // Step 2: All done!
  return 0;

}

int compute_displacements_and_build_next_eigenframe( CUDAstate_struct *restrict CUDAstate,
                                                     REAL *restrict displacement ) {

  // Step 1: Compute the FFT of the new image
  FFT_EXECUTE_DFT(CUDAstate->fft2_plan,
                  CUDAstate->squared_new_image_time,
                  CUDAstate->squared_new_image_freq,
                  CUFFT_FORWARD);

  // Step 2: Compute image product target * src^{*}
  const int Nh   = CUDAstate->N_horizontal;
  const int Nv   = CUDAstate->N_vertical;
  const int NhNv = Nh*Nv;
  element_wise_multiplication_conj(Nh,Nv,NhNv,CUDAstate->squared_new_image_freq,
                                   CUDAstate->reciprocal_eigenframe_freq,
                                   CUDAstate->aux_array1);

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFT_EXECUTE_DFT(CUDAstate->fft2_plan,
                  CUDAstate->aux_array1,
                  CUDAstate->aux_array2,
                  CUFFT_INVERSE);

  // Step 4: Full-pixel estimate of the cross correlation maxima
  compute_absolute_value(Nh, Nv, NhNv, CUDAstate->aux_array2, CUDAstate->aux_array_real);
  const int idx_min = find_minima(CUDAstate->cublasHandle,NhNv,CUDAstate->aux_array_real);
  // Flattened indices are computed using idx = i + Nh*j, so that
  // j = idx/Nh - i/Nh = idx/Nh,
  // because integer division i/Nh = 0 since i < Nh.
  // Once j is known, we can compute i using
  // i = idx - Nh*j
  const int j_min = idx_min/Nh;
  const int i_min = idx_min - j_min*Nh;
  displacement[0] = (REAL)i_min;
  displacement[1] = (REAL)j_min;

  // Step 5: Compute midpoint
  const int midpoint[2] = {Nh/2,Nv/2};

  // Step 6: If the displacement is larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacement[0] > midpoint[0] ) displacement[0] -= Nh;
  if( displacement[1] > midpoint[1] ) displacement[1] -= Nv;

  // Step 7: Upsample the image around the maxima and
  //         find new maxima with sub-pixel precision
  // Note: aux_array1 stores the image product, while
  //       aux_array2 and aux_array3 can be used for
  //       any purpose
  const REAL upsample_factor = CUDAstate->upsample_factor;
  if( (int)(upsample_factor+0.5) > 1 )
    get_subpixel_displacement_by_upsampling(CUDAstate,displacement);

  // Step 8: Loop over all images in the dataset, reverse shift
  //         them (in Fourier space) and add them to the eigenframe
  compute_reverse_shift_2D(Nh, Nv, displacement,
                           CUDAstate->aux_array1,
                           CUDAstate->aux_array2,
                           CUDAstate->aux_array3);

  // Step 9: Now shift the image and add it to the eigenframe
  shift_image_add_to_eigenframe(CUDAstate);

  return 0;
}
