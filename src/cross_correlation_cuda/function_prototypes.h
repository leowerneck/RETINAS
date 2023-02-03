#ifndef FUNCTION_PROTOTYPES_H_
#define FUNCTION_PROTOTYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes

// This function is implemented in image_analysis.cu
__host__
int set_zeroth_eigenframe( CUDAstate_struct *restrict CUDAstate );

// This function is implemented in image_analysis.cu
__host__
int cross_correlate_and_build_next_eigenframe( CUDAstate_struct *restrict CUDAstate,
                                               REAL *restrict displacement );

// This function is implemented in get_subpixel_displacement_by_upsampling.cu
__host__
void get_subpixel_displacement_by_upsampling( CUDAstate_struct *restrict CUDAstate,
                                              REAL *restrict displacement );

// This function is implemented in element_wise_multiplication_conj.cu
__global__
void element_wise_multiplication_conj_GPU(const int n, const COMPLEX *restrict A, const COMPLEX *restrict B, COMPLEX *restrict C);

// This function is implemented in element_wise_multiplication_conj.cu
__host__
void element_wise_multiplication_conj(const int m, const int n, const int mn,
                                      const COMPLEX *restrict A, const COMPLEX *restrict B, COMPLEX *restrict C);

// This function is implemented in find_maxima.cu
__host__
int find_maxima(cublasHandle_t cublasHandle, const int n, REAL *restrict z);

// This function is implemented in initialize_finalize_CUDAstate.cu
__host__
CUDAstate_struct *initialize_CUDAstate( const int N_horizontal,
                                        const int N_vertical,
                                        const REAL upsample_factor,
                                        const REAL A0,
                                        const REAL B1 );

// This function is implemented in initialize_finalize_CUDAstate.cu
__host__
int finalize_CUDAstate( CUDAstate_struct *restrict CUDAstate );

// This function is implemented in preprocess_cross_correlation_data.cu
__host__
REAL typecast_and_return_brightness( const uint16_t *restrict input_array,
                                     CUDAstate_struct *restrict CUDAstate );

// This function is implemented in preprocess_cross_correlation_data.cu
__host__
REAL typecast_rebin_4x4_and_return_brightness( const uint16_t *restrict input_array,
                                               CUDAstate_struct *restrict CUDAstate );

// This function is implemented in compute_reverse_shift_2D
__host__
void compute_reverse_shift_2D(const int N_horizontal,
                              const int N_vertical,
                              const REAL *restrict displacements,
                              COMPLEX *restrict horizontal_shifts,
                              COMPLEX *restrict vertical_shifts,
                              COMPLEX *restrict shift2D);

// This function is implemented in complex_conjugate_2d.cu
__host__
void complex_conjugate_2d(
    const int m,
    const int n,
    COMPLEX *restrict z );

// This function is implemented in element_wise_multiplication_conj.cu
__host__
void element_wise_multiplication_conj(const int m,
                                      const int n,
                                      const int mn,
                                      const COMPLEX *restrict A,
                                      const COMPLEX *restrict B,
                                      COMPLEX *restrict C);

// This function is implemented in matrix_multiplication.cu
__host__
void complex_matrix_multiply_tt(cublasHandle_t cublasHandle,
                                const int m,
                                const int p,
                                const int n,
                                const COMPLEX *restrict A,
                                const COMPLEX *restrict B,
                                COMPLEX *restrict C);

// This function is implemented in matrix_multiplication.cu
__host__
void complex_matrix_multiply(cublasHandle_t cublasHandle,
                             const int m,
                             const int p,
                             const int n,
                             const COMPLEX *restrict A,
                             const COMPLEX *restrict B,
                             COMPLEX *restrict C);

// This function is implemented in shift_image_add_to_eigenframe.cu
__host__
void shift_image_add_to_eigenframe(CUDAstate_struct *restrict CUDAstate);

// This function is implemented in compute_kernels.cu
__host__
void compute_horizontal_kernel(const REAL *restrict sample_region_offset, CUDAstate_struct *restrict CUDAstate);

// This function is implemented in compute_kernels.cu
__host__
void compute_vertical_kernel(const REAL *restrict sample_region_offset, CUDAstate_struct *restrict CUDAstate);

// This function is implemented in absolute_value_2d.cu
__host__
void absolute_value_2d(
    const int m,
    const int n,
    const COMPLEX *restrict z,
    REAL *restrict x );

// Inline function to compute the exponential of a complex number
__host__ __device__
static inline
COMPLEX CEXP(COMPLEX z) {
  // z = a + ib
  // exp(z) = exp(a)exp(ib) = exp(a)( cos(b) + isin(b) )
  REAL a    = z.x; // Real part of z
  REAL b    = z.y; // Imag part of z
  REAL expa = EXP(a);
  return MAKE_COMPLEX(expa*COS(b),expa*SIN(b));
}

__host__
int dump_eigenframe( CUDAstate_struct *restrict CUDAstate );

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FUNCTION_PROTOTYPES_H_
