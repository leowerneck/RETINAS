#include "retina.h"

/*
 *  Function: complex_matrix_multiply
 *  Author  : Leo Werneck
 *
 *  Performs the matrix multiplication C = A.B.
 *  This function uses CBLAS.
 *
 *  Inputs
 *  ------
 *    m : in
 *      Common dimension of matrices A and C.
 *
 *    p : in
 *      Common dimension of matrices A and B.
 *
 *    n : in
 *      Common dimension of matrices B and C.
 *
 *    A : in
 *      Flattened matrix of dimension m x p.
 *
 *    B : in
 *      Flattened matrix of dimension p x n.
 *
 *    C : out
 *      Flattened matrix of dimension m x n. Stores the result.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
static inline
void complex_matrix_multiply(
    const int m,
    const int p,
    const int n,
    const void *restrict A,
    const void *restrict B,
    void *restrict C ) {

  const REAL a[2] = {1.0,0.0};
  const REAL b[2] = {0.0,0.0};
  CBLAS_GEMM(CblasColMajor,CblasNoTrans,CblasNoTrans,
             m,n,p,a,A,m,B,p,b,C,m);
}

/*
 *  Function: complex_matrix_multiply_tt
 *  Author  : Leo Werneck
 *
 *  Performs the matrix multiplication C = (A.B)^T = BT.AT, where
 *  AT/BT is the transpose of A/B. This function uses CBLAS.
 *
 *  Inputs
 *  ------
 *    m : in
 *      Common dimension of matrices A and C.
 *
 *    p : in
 *      Common dimension of matrices A and B.
 *
 *    n : in
 *      Common dimension of matrices B and C.
 *
 *    A : in
 *      Flattened matrix of dimension m x p. AT is p x m.
 *
 *    B : in
 *      Flattened matrix of dimension p x n. BT is n x p.
 *
 *    C : out
 *      Flattened matrix of dimension n x m. Stores the result.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
static inline
void complex_matrix_multiply_tt(
    const int m,
    const int p,
    const int n,
    const void *restrict A,
    const void *restrict B,
    void *restrict C ) {

  const REAL a[2] = {1.0,0.0};
  const REAL b[2] = {0.0,0.0};
  CBLAS_GEMM(CblasColMajor,CblasTrans,CblasTrans,
             n,m,p,a,B,p,A,m,b,C,m);
}

/*
 *  Function: upsample_around_displacements
 *  Author  : Leo Werneck
 *
 *  Upsample the region around displacements and recompute
 *  them with subpixel precision.
 *
 *  Inputs
 *  ------
 *    state : in/out
 *      The state object (see retina.h).
 *
 *    displacements : in/out
 *      Horizontal and vertical displacements. Stores the result.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void upsample_around_displacements(
    state_struct *restrict state,
    REAL *restrict displacements ) {

  // Step 1: Adjust the displacements based on the upsample factor
  const REAL upsample_factor = state->upsample_factor;
  for(int i=0;i<2;i++) displacements[i] = ROUND(displacements[i] * upsample_factor)/upsample_factor;

  // Step 2: Set size of upsampled region
  const REAL upsampled_region_size = CEIL(upsample_factor * 1.5);

  // Step 3: Center of the output array at dftshift+1
  const REAL dftshift = round_towards_zero(upsampled_region_size / 2.0);

  // Step 4: Compute upsample region offset
  REAL sample_region_offset[2];
  for(int i=0;i<2;i++) sample_region_offset[i] = dftshift - displacements[i]*upsample_factor;

  // Step 5: Useful constant
  const COMPLEX im2pi = 0.0 - I * 2.0 * M_PI;

  // Step 6: Upsampled size
  const int S = (int)upsampled_region_size;

  // Step 7: Compute the horizontal kernel
  const int N_horizontal     = state->N_horizontal;
  const int N_vertical       = state->N_vertical;
  const int nhalf_horizontal = FLOOR((N_horizontal-1)/2.0)+1;
  const int aux_horizontal   = nhalf_horizontal+FLOOR(N_horizontal/2.0);
  const REAL norm_horizontal = 1.0/(N_horizontal*upsample_factor);
  for(int i_h=0;i_h<nhalf_horizontal;i_h++) {
    for(int i_s=0;i_s<S;i_s++) {
      const REAL fft_freq = i_h * norm_horizontal;
      const REAL kernel_l = (i_s - sample_region_offset[0])*fft_freq;
      state->aux_array2[i_s+S*i_h] = CEXP(im2pi * kernel_l);
    }
  }
  for(int i_h=nhalf_horizontal;i_h<N_horizontal;i_h++) {
    for(int i_s=0;i_s<S;i_s++) {
      const REAL fft_freq = (i_h - aux_horizontal) * norm_horizontal;
      const REAL kernel_l = (i_s - sample_region_offset[0])*fft_freq;
      state->aux_array2[i_s+S*i_h] = CEXP(im2pi * kernel_l);
    }
  }

  // Step 8: Contract the horizontal kernel with the conjugate of the image product
  for(int i=0;i<N_horizontal*N_vertical;i++)
    state->aux_array1[i] = CONJ(state->aux_array1[i]);

  // Note: aux_array1 contains the complex conjugate of the image product,
  //       aux_array2 contains the horizontal kernel, and
  //       aux_array3 will contain the matrix product of aux_array2 and aux_array1.
  complex_matrix_multiply(S, N_horizontal, N_vertical,
                          state->aux_array2, state->aux_array1, state->aux_array3);

  // Step 9: Compute the vertical kernel (Nv x S)
  const int nhalf_vertical = FLOOR((N_vertical-1)/2.0)+1;
  const int aux_vertical   = nhalf_vertical+FLOOR(N_vertical/2.0);
  const REAL norm_vertical = 1.0/(N_vertical*upsample_factor);
  for(int i_s=0;i_s<S;i_s++) {
    for(int i_v=0;i_v<nhalf_vertical;i_v++) {
      const REAL fft_freq = i_v * norm_vertical;
      const REAL kernel_l = (i_s - sample_region_offset[1])*fft_freq;
      state->aux_array2[i_v+N_vertical*i_s] = CEXP(im2pi * kernel_l);
    }
    for(int i_v=nhalf_vertical;i_v<N_vertical;i_v++) {
      const REAL fft_freq = (i_v - aux_vertical) * norm_vertical;
      const REAL kernel_l = (i_s - sample_region_offset[1])*fft_freq;
      state->aux_array2[i_v+N_vertical*i_s] = CEXP(im2pi * kernel_l);
    }
  }

  // Step 10: Now contract the result of Step 8 with the vertical kernel to get the upsampled image
  // Note: aux_array1 will contains the upsampled image,
  //       aux_array2 contains the vertical kernel, and
  //       aux_array3 is the same as in Step 8.
  complex_matrix_multiply_tt(S, N_vertical, S,
                             state->aux_array3, state->aux_array2, state->aux_array1);
}
