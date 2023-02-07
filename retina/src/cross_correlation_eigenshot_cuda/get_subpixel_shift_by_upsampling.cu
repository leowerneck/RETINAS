#include "image_analysis.h"

// This function is required by the upsampling algorithm.
// It rounds a number to the nearest integer towards zero, e.g.
// +2.9  => +2
// +3.14 => +3
// -3.14 => -3
// -2.9  => -2
// This function was validated against the NumPy function fix.
int round_towards_zero(const REAL x) {
  if( x > 0.0 )
    return FLOOR(x);
  else
    return CEIL(x);
}

/*
 *
 * (c) 2021 Leo Werneck
 *
 * Perform upsampling on a 2D image. This is a C translation
 * of the _upsample_dft() function from Scikit-Image.
 */
extern "C" __host__
void get_subpixel_displacement_by_upsampling(CUDAstate_struct *restrict CUDAstate,
                                             REAL *restrict displacement) {

  // Note: At the beginning of this function,
  //       aux_array1 stores the image product, while
  //       aux_array2 and aux_array3 can be used for
  //       any purpose

  // Step 1: Set basic variables
  const int Nh   = CUDAstate->N_horizontal;
  const int Nv   = CUDAstate->N_vertical;
  const int NhNv = Nh*Nv;
  const REAL upsample_factor = CUDAstate->upsample_factor;

  // Step 2: Adjust the displacement based on the upsample factor
  for(int i=0;i<2;i++) displacement[i] = ROUND(displacement[i] * upsample_factor)/upsample_factor;

  // Step 3: Set size of upsampled region
  const REAL upsampled_region_size = CEIL(upsample_factor * 1.5);

  // Step 4: Center of the output array at dftshift+1
  const REAL dftshift = round_towards_zero(upsampled_region_size / 2.0);

  // Step 5: Compute upsample region offset
  REAL sample_region_offset[2];
  for(int i=0;i<2;i++) sample_region_offset[i] = dftshift - displacement[i]*upsample_factor;

  // Step 6: Upsampled size
  const int S  = (int)upsampled_region_size;
  const int SS = S*S;

  // Step 7: Compute the horizontal kernel
  compute_horizontal_kernel(sample_region_offset, CUDAstate);

  // Step 8: Contract the horizontal kernel with the conjugate of the image product
  complex_conjugate(Nh,Nv,NhNv,CUDAstate->aux_array1);
  // Note: aux_array1 contains the complex conjugate of the image product,
  //       aux_array2 contains the horizontal kernel, and
  //       aux_array3 will contain the matrix product of aux_array2 and aux_array1.
  complex_matrix_multiply(CUDAstate->cublasHandle,S,Nh,Nv,CUDAstate->aux_array2,CUDAstate->aux_array1,CUDAstate->aux_array3);

  // Step 9: Compute the vertical kernel
  compute_vertical_kernel(sample_region_offset, CUDAstate);

  // Step 10: Now contract the result of Step 8 with the vertical kernel to get the upsampled image
  // Note: aux_array1 will contains the upsampled image,
  //       aux_array2 contains the vertical kernel, and
  //       aux_array3 is the same as in Step 8.
  complex_matrix_multiply_tt(CUDAstate->cublasHandle,S,Nv,S,CUDAstate->aux_array3,CUDAstate->aux_array2,CUDAstate->aux_array1);

  // Step 10: Get maximum of upsampled image
  compute_absolute_value(S, S, SS, CUDAstate->aux_array1, CUDAstate->aux_array_real);
  const int idx_min = find_minima(CUDAstate->cublasHandle, SS, CUDAstate->aux_array_real);
  const int i_min   = idx_min/S;
  const int j_min   = idx_min - i_min*S;

  // Step 11: Update the displacement
  displacement[0] += ((REAL)i_min - dftshift)/upsample_factor;
  displacement[1] += ((REAL)j_min - dftshift)/upsample_factor;
}
