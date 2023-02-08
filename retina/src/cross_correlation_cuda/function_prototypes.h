#ifndef FUNCTION_PROTOTYPES_H_
#define FUNCTION_PROTOTYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

// Function prototypes
// This function is implemented in state_initialize.cu
__host__
state_struct *state_initialize(
    const int N_horizontal,
    const int N_vertical,
    const REAL upsample_factor,
    const REAL A0,
    const REAL B1,
    const REAL shift );

// This function is implemented in state_finalize.cu
__host__
void state_finalize( state_struct *restrict state );

// This function is implemented in typecast_input_image_and_compute_brightness.cu
__host__
REAL typecast_input_image_and_compute_brightness(
    const uint16_t *restrict input_array,
    state_struct *restrict state );

// This function is implemented in typecast_input_image_and_compute_brightness_shot_noise.cu
__host__
REAL typecast_input_image_and_compute_brightness_shot_noise(
    const uint16_t *restrict input_array,
    state_struct *restrict state );

// This function is implemented in set_zeroth_eigenframe.cu
__host__
void set_zeroth_eigenframe( state_struct *restrict state );

// This function is implemented in cross_correlate_ref_and_new_images.cu
__host__
void cross_correlate_ref_and_new_images( state_struct *restrict state );

// This function is implemented in displacements_full_pixel_estimate.cu
__host__
void displacements_full_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in displacements_full_pixel_estimate_shot_noise.cu
__host__
void displacements_full_pixel_estimate_shot_noise(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in upsample_around_displacements.cu
__host__
void upsample_around_displacements(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in displacements_sub_pixel_estimate.cu
__host__
void displacements_sub_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in displacements_sub_pixel_estimate_shot_noise.cu
__host__
void displacements_sub_pixel_estimate_shot_noise(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in build_next_eigenframe.cu
__host__
void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state );

// This function is implemented in compute_reverse_shift_matrix.cu
__host__
void compute_reverse_shift_matrix(
    const int N_horizontal,
    const int N_vertical,
    const REAL *restrict displacements,
    COMPLEX *restrict horizontal_shifts,
    COMPLEX *restrict vertical_shifts,
    COMPLEX *restrict shift2D );

// This function is implemented in compute_displacements_and_build_next_eigenframe.cu
__host__
void compute_displacements_and_build_next_eigenframe(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in compute_kernels.cu
__host__
void compute_horizontal_kernel(
    const REAL *restrict sample_region_offset,
    state_struct *restrict state );

// This function is implemented in compute_kernels.cu
__host__
void compute_vertical_kernel(
    const REAL *restrict sample_region_offset,
    state_struct *restrict state );

// This function is implemented in absolute_value_2d.cu
__host__
void absolute_value_2d(
    const int m,
    const int n,
    const COMPLEX *restrict z,
    REAL *restrict x );

// This function is implemented in find_maxima_minima.cu
__host__
int find_maxima(
    cublasHandle_t h,
    const int n,
    REAL *restrict x );

// This function is implemented in find_maxima_minima.cu
__host__
int find_minima(
    cublasHandle_t h,
    const int n,
    REAL *restrict x );

// This function is implemented in complex_conjugate_2d.cu
__host__
void complex_conjugate_2d(
    const int m,
    const int n,
    COMPLEX *restrict z );

// This function is implemented in element_wise_multiplication_conj_2d.cu
__host__
void element_wise_multiplication_conj_2d(
    const int m,
    const int n,
    const COMPLEX *restrict A,
    const COMPLEX *restrict B,
    COMPLEX *restrict C );

// This function is implemented in get_eigenframe.cu
__host__
void get_eigenframe(
    state_struct *restrict state,
    REAL *restrict eigenframe_time );

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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FUNCTION_PROTOTYPES_H_
