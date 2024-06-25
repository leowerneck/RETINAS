#include "retinas.h"

__host__ static inline
void complex_matrix_multiply(
    cublasHandle_t h,
    const int m,
    const int p,
    const int n,
    const COMPLEX *restrict A,
    const COMPLEX *restrict B,
    COMPLEX *restrict C ) {
  /*
   *  Performs the matrix multiplication C = A.B. This function
   *  uses cuBLAS.
   *
   *  Inputs
   *  ------
   *    h : cuBLAS handle object.
   *    m : Common dimension of matrices A and C.
   *    p : Common dimension of matrices A and B.
   *    n : Common dimension of matrices B and C.
   *    A : Flattened matrix of dimension m x p.
   *    B : Flattened matrix of dimension p x n.
   *    C : Flattened matrix of dimension m x n. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const COMPLEX alpha = MAKE_COMPLEX(1.0, 0.0);
  const COMPLEX beta  = MAKE_COMPLEX(0.0, 0.0);
  CUBLASCGEMM(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, p,
              &alpha, A, m, B, p, &beta, C, m);
}

__host__ static inline
void complex_matrix_multiply_tt(
    cublasHandle_t h,
    const int m,
    const int p,
    const int n,
    const COMPLEX *restrict A,
    const COMPLEX *restrict B,
    COMPLEX *restrict C ) {
  /*
   *  Performs the matrix multiplication C = (A.B)^T = BT.AT, where
   *  AT/BT is the transpose of A/B. This function uses cuBLAS.
   *
   *  Inputs
   *  ------
   *    h : cuBLAS handle object.
   *    m : Common dimension of matrices A and C.
   *    p : Common dimension of matrices A and B.
   *    n : Common dimension of matrices B and C.
   *    A : Flattened matrix of dimension m x p. AT is p x m.
   *    B : Flattened matrix of dimension p x n. BT is n x p.
   *    C : Flattened matrix of dimension n x m. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   */

  const COMPLEX alpha = MAKE_COMPLEX(1.0,0.0);
  const COMPLEX beta  = MAKE_COMPLEX(0.0,0.0);
  CUBLASCGEMM(h, CUBLAS_OP_T, CUBLAS_OP_T, n, m, p,
              &alpha, B, p, A, m, &beta, C, m);
}

extern "C" __host__
void upsample_around_displacements(
    state_struct *restrict state,
    REAL *restrict displacements ) {
  /*
   *  Upsample the region around displacements and recompute
   *  them with subpixel precision.
   *
   *  Inputs
   *  ------
   *    state         : The CUDA state object containing all required data.
   *    displacements : Array of horizontal and vertical displacements. Stores the result.
   *
   *  Returns
   *  -------
   *    Nothing.
   *
   *  Note
   *  ----
   *    Not all pointers inside the state object are unique. Keep the following
   *    information in mind when dealing with the pointers in this function:
   *
   *      state->image_product     == state->aux_array1
   *      state->upsampled_image   == state->aux_array1
   *      state->horizontal_kernel == state->aux_array2
   *      state->vertical_kernel   == state->aux_array2
   *
   *    For more information, see retinas.h and state_initialize.cu.
   */

  // Step 1: Set basic variables
  const int Nh = state->N_horizontal;
  const int Nv = state->N_vertical;
  const REAL upsample_factor = state->upsample_factor;

  // Step 2: Adjust the displacements based on the upsample factor
  for(int i=0;i<2;i++)
    displacements[i] = ROUND(displacements[i] * upsample_factor)/upsample_factor;

  // Step 3: Set size of upsampled region
  const REAL upsampled_region_size = CEIL(upsample_factor * 1.5);

  // Step 4: Center of the output array at dftshift+1
  const REAL dftshift = round_towards_zero(upsampled_region_size / 2.0);

  // Step 5: Compute upsample region offset
  REAL sample_region_offset[2];
  for(int i=0;i<2;i++)
    sample_region_offset[i] = dftshift - displacements[i]*upsample_factor;

  // Step 6: Upsampled size
  const int S  = (int)upsampled_region_size;

  // Step 7: Compute the horizontal kernel (set to aux_array2)
  compute_horizontal_kernel(sample_region_offset, state);

  // Step 8: Contract the horizontal kernel with the conjugate of the image product
  complex_conjugate_2d(Nh, Nv, state->image_product);
  complex_matrix_multiply(state->cublasHandle,
                          S,
                          Nh,
                          Nv,
                          state->horizontal_kernel, // Size  S x Nh
                          state->image_product,     // Size Nh x Nv
                          state->partial_product);  // Size  S x Nv

  // Step 9: Compute the vertical kernel (set in aux_array2)
  compute_vertical_kernel(sample_region_offset, state);

  // Step 10: Now contract the result of Step 8 with the vertical kernel to get the upsampled image
  complex_matrix_multiply_tt(state->cublasHandle,
                             S,
                             Nv,
                             S,
                             state->partial_product,  // Size  S x Nv
                             state->vertical_kernel,  // Size Nv x S
                             state->upsampled_image); // Size  S x S
}
