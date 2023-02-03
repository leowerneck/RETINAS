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
    const REAL B1 );

// This function is implemented in state_finalize.cu
__host__
void state_finalize( state_struct *restrict state );

// This function is implemented in typecast_and_return_brightness.cu
__host__
REAL typecast_input_image_and_compute_brightness(
    const uint16_t *restrict input_array,
    state_struct *restrict state );

// This function is implemented in set_zeroth_eigenframe.cu
__host__
void set_zeroth_eigenframe( state_struct *restrict state );

// This function is implemented in cross_correlate_and_compute_displacements.cu
__host__
void cross_correlate_and_compute_displacements(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in upsample_and_compute_subpixel_displacements.cu
__host__
void upsample_and_compute_subpixel_displacements(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in absolute_value_2d.cu
__host__
void absolute_value_2d(
    const int m,
    const int n,
    const COMPLEX *restrict z,
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

static inline
void info(const char *format, ...) {
  /*
   *  Slightly modified printf which appends the
   *  code name to the beginning of the message.
   */
  printf("(RETINA) ");
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FUNCTION_PROTOTYPES_H_
