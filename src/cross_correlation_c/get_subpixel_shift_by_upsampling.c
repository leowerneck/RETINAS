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

// C = A.B, where:
// A is m x p
// B is p x n
// C is m x n
void complex_matrix_multiply(const int m,
                             const int p,
                             const int n,
                             const void *restrict A,
                             const void *restrict B,
                             void *restrict C) {
  const REAL a[2] = {1.0,0.0};
  const REAL b[2] = {0.0,0.0};
  CBLAS_GEMM(CblasColMajor,CblasNoTrans,CblasNoTrans,
             m,n,p,a,A,m,B,p,b,C,m);
}

// C = (A.B)^{T} = B^{T}.A^{T}, where:
// A is m x p, A^{T} is p x m
// B is p x n, B^{T} is n x p
// C is n x m
void complex_matrix_multiply_tt(const int m,
                                const int p,
                                const int n,
                                const void *restrict A,
                                const void *restrict B,
                                void *restrict C) {
  const REAL a[2] = {1.0,0.0};
  const REAL b[2] = {0.0,0.0};
  CBLAS_GEMM(CblasColMajor,CblasTrans,CblasTrans,
             n,m,p,a,B,p,A,m,b,C,m);
}

/*
 *
 * (c) 2021 Leo Werneck
 *
 * Perform upsampling on a 2D image. This is a C translation
 * of the _upsample_dft() function from Scikit-Image.
 */
void get_subpixel_displacement_by_upsampling(const int N_horizontal,
                                             const int N_vertical,
                                             const REAL upsample_factor,
                                             FFTW_PLAN ifft2_plan,
                                             COMPLEX *restrict aux_array1,
                                             COMPLEX *restrict aux_array2,
                                             COMPLEX *restrict aux_array3,
                                             REAL *restrict displacement) {

  // Note: At the beginning of this function,
  //       aux_array1 stores the image product, while
  //       aux_array2 and aux_array3 can be used for
  //       any purpose

  // Step 1: Adjust the displacement based on the upsample factor
  for(int i=0;i<2;i++) displacement[i] = ROUND(displacement[i] * upsample_factor)/upsample_factor;

  // Step 2: Set size of upsampled region
  const REAL upsampled_region_size = CEIL(upsample_factor * 1.5);

  // Step 3: Center of the output array at dftshift+1
  const REAL dftshift = round_towards_zero(upsampled_region_size / 2.0);

  // Step 4: Compute upsample region offset
  REAL sample_region_offset[2];
  for(int i=0;i<2;i++) sample_region_offset[i] = dftshift - displacement[i]*upsample_factor;

  // Step 5: Useful constant
  const COMPLEX im2pi = 0.0 - I * 2.0 * M_PI;

  // Step 6: Upsampled size
  const int S = (int)upsampled_region_size;

  // Step 7: Compute the horizontal kernel
  const int nhalf_horizontal = FLOOR((N_horizontal-1)/2.0)+1;
  const REAL norm_horizontal = 1.0/(N_horizontal*upsample_factor);
  REAL fft_freq;
  for(int i_s=0;i_s<S;i_s++) {
    for(int i_h=0;i_h<N_horizontal;i_h++) {
      if( i_h < nhalf_horizontal ) {
        fft_freq = i_h * norm_horizontal;
      }
      else {
        fft_freq = (-FLOOR(N_horizontal/2.0)+i_h-nhalf_horizontal) * norm_horizontal;
      }
      const REAL kernel_l = (i_s - sample_region_offset[0])*fft_freq;
      aux_array2[i_s+S*i_h] = CEXP(im2pi * kernel_l);
    }
  }

  // Step 8: Contract the horizontal kernel with the conjugate of the image product
  for(int i=0;i<N_horizontal*N_vertical;i++) {
    aux_array1[i] = CONJ(aux_array1[i]);
  }
  // Note: aux_array1 contains the complex conjugate of the image product,
  //       aux_array2 contains the horizontal kernel, and
  //       aux_array3 will contain the matrix product of aux_array2 and aux_array1.
  complex_matrix_multiply(S,N_horizontal,N_vertical,aux_array2,aux_array1,aux_array3);

  // Step 9: Compute the vertical kernel
  const int nhalf_vertical = FLOOR((N_vertical-1)/2.0)+1;
  const REAL norm_vertical = 1.0/(N_vertical*upsample_factor);
  for(int i_s=0;i_s<S;i_s++) {
    for(int i_v=0;i_v<N_vertical;i_v++) {
      if( i_v < nhalf_vertical ) {
        fft_freq = i_v * norm_vertical;
      }
      else {
        fft_freq = (-FLOOR(N_vertical/2.0)+i_v-nhalf_vertical) * norm_vertical;
      }
      const REAL kernel_l = (i_s - sample_region_offset[1])*fft_freq;
      aux_array2[i_v+N_vertical*i_s] = CEXP(im2pi * kernel_l);
    }
  }

  // Step 10: Now contract the result of Step 8 with the vertical kernel to get the upsampled image
  // Note: aux_array1 will contains the upsampled image,
  //       aux_array2 contains the vertical kernel, and
  //       aux_array3 is the same as in Step 8.
  complex_matrix_multiply_tt(S,N_vertical,S,aux_array3,aux_array2,aux_array1);

  // Step 10: Get maximum of upsampled image
  int i_max=0,j_max=0;
  REAL cc_max = -1.0;
  for(int j_s=0;j_s<S;j_s++) {
    for(int i_s=0;i_s<S;i_s++) {
      const REAL cc = CABS(aux_array1[i_s + S*j_s]);
      if( cc > cc_max ) {
        cc_max = cc;
        // FIXME: Why do I need to flip?
        // Answer: It has to do with the FFT index ordering. I think
        // using Ny,Nx instead of Nx,Ny fixes this, but needs testing.
        i_max  = j_s;
        j_max  = i_s;
      }
    }
  }

  // Step 11: Update the displacement
  displacement[0] += ((REAL)i_max - dftshift)/upsample_factor;
  displacement[1] += ((REAL)j_max - dftshift)/upsample_factor;
}
