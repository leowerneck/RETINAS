// NRPy+ basic definitions, automatically generated from outC_NRPy_basic_defines_h_dict within outputC,
//    and populated within NRPy+ modules. DO NOT EDIT THIS FILE BY HAND.



//********************************************
// Basic definitions for module outputC:

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // "string.h Needed for strncmp, etc.
#include <stdint.h> // "stdint.h" Needed for Windows GCC 6.x compatibility, and for int8_t
#include <stdarg.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884L
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039L
#endif

#ifdef __cplusplus
#define restrict __restrict__
#endif
//********************************************


//********************************************
// Basic definitions for module Big_G:

// Macros to compute minimum and maximum
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )
#define MAX(a,b) ( (a) > (b) ? (a) : (b) )

// C complex type
#include <complex.h>

// FFTW library
#include <fftw3.h>

// CBLAS library
#include <cblas.h>

// Precision macros
#define SINGLE 0
#define DOUBLE 1
#define PRECISION DOUBLE

// Useful macros
#if PRECISION == SINGLE
#  define REAL float
#  define ROUND roundf
#  define FLOOR floorf
#  define CEIL ceilf
#  define COMPLEX fftwf_complex
#  define CREAL crealf
#  define CIMAG cimagf
#  define CABS cabsf
#  define CONJ conjf
#  define CEXP cexpf
#  define COMPLEX fftwf_complex
#  define FFTW_ALLOC_REAL fftwf_alloc_real
#  define FFTW_ALLOC_COMPLEX fftwf_alloc_complex
#  define FFTW_FREE fftwf_free
#  define FFTW_PLAN_DFT_2D fftwf_plan_dft_2d
#  define FFTW_PLAN_DFT_R2C_2D fftwf_plan_dft_r2c_2d
#  define FFTW_PLAN_DFT_C2R_2D fftwf_plan_dft_c2r_2d
#  define FFTW_PLAN fftwf_plan
#  define FFTW_DESTROY_PLAN fftwf_destroy_plan
#  define FFTW_EXECUTE fftwf_execute
#  define FFTW_EXECUTE_DFT fftwf_execute_dft
#  define FFTW_EXECUTE_DFT_R2C(r, c) fftwf_execute_dft_r2c(state->fftf, r, c)
#  define FFTW_EXECUTE_DFT_C2R(c, r) fftwf_execute_dft_c2r(state->ffti, c, r)
#  define FFTW_CLEANUP fftwf_cleanup
#  define CBLAS_GEMM cblas_cgemm
#  define CBLAS_IAMAX_REAL cblas_isamax
#  define CBLAS_IAMAX_COMPLEX cblas_icamax
#else
#  define REAL double
#  define ROUND round
#  define FLOOR floor
#  define CEIL ceil
#  define COMPLEX fftw_complex
#  define CREAL creal
#  define CIMAG cimag
#  define CABS cabs
#  define CONJ conj
#  define CEXP cexp
#  define COMPLEX fftw_complex
#  define FFTW_ALLOC_REAL fftw_alloc_real
#  define FFTW_ALLOC_COMPLEX fftw_alloc_complex
#  define FFTW_FREE fftw_free
#  define FFTW_PLAN_DFT_2D fftw_plan_dft_2d
#  define FFTW_PLAN_DFT_R2C_2D fftw_plan_dft_r2c_2d
#  define FFTW_PLAN_DFT_C2R_2D fftw_plan_dft_c2r_2d
#  define FFTW_PLAN fftw_plan
#  define FFTW_DESTROY_PLAN fftw_destroy_plan
#  define FFTW_EXECUTE fftw_execute
#  define FFTW_EXECUTE_DFT fftw_execute_dft
#  define FFTW_EXECUTE_DFT_R2C(r, c) fftw_execute_dft_r2c(state->fftf, r, c)
#  define FFTW_EXECUTE_DFT_C2R(c, r) fftw_execute_dft_c2r(state->ffti, c, r)
#  define FFTW_CLEANUP fftw_cleanup
#  define CBLAS_GEMM cblas_zgemm
#  define CBLAS_IAMAX_REAL cblas_idamax
#  define CBLAS_IAMAX_COMPLEX cblas_izamax
#endif

// Image analysis parameter struct
typedef struct state_struct {
  bool shot_noise_method;
  int N_horizontal, N_vertical, aux_size;
  REAL upsample_factor, A0, B1, shift;
  FFTW_PLAN fft2_plan, ifft2_plan;
  COMPLEX *restrict aux_array1;
  COMPLEX *restrict aux_array2;
  COMPLEX *restrict aux_array3;
  COMPLEX *restrict reciprocal_new_image_time;
  COMPLEX *restrict new_image_time;
  COMPLEX *restrict new_image_freq;
  COMPLEX *restrict ref_image_freq;

  // Testing
  int N_aux;
  REAL    *restrict Itime;
  COMPLEX *restrict Ifreq;
  COMPLEX *restrict Efreq;
  COMPLEX *restrict image_product;
  REAL    *restrict cross_correlation;
  FFTW_PLAN fftf, ffti;
} state_struct;

// .---------------------.
// | Function prototypes |
// .---------------------.
// This function is implemented in state_initialize.c
state_struct *state_initialize(
    const int N_horizontal,
    const int N_vertical,
    const REAL upsample_factor,
    const REAL A0,
    const REAL B1,
    const bool shot_noise_method,
    const REAL shift );

// This function is implemented in state_finalize.c
void state_finalize( state_struct *restrict state );

// This function is implemented in typecast_input_image_and_compute_brightness.c
REAL typecast_input_image_and_compute_brightness(
    const uint16_t *restrict input_array,
    state_struct *restrict state );

// This function is implemented in typecast_input_image_rebin_4x4_and_compute_brightness.c
REAL typecast_input_image_rebin_4x4_and_compute_brightness(
    const uint16_t *restrict input_array,
    state_struct *restrict state );

// This function is implemented in set_zeroth_eigenframe.c
void set_zeroth_eigenframe( state_struct *restrict state );

// This function is implemented in cross_correlate_ref_and_new_images.c
void cross_correlate_ref_and_new_images( state_struct *restrict state );

// This function is implemented in displacements_full_pixel_estimate.c
void displacements_full_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in displacements_sub_pixel_estimate.c
void displacements_sub_pixel_estimate(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in upsample_around_displacements.c
void upsample_around_displacements(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in compute_reverse_shift_matrix.c
void compute_reverse_shift_matrix(
    const int N_horizontal,
    const int N_vertical,
    const REAL *restrict displacements,
    COMPLEX *restrict aux_array1,
    COMPLEX *restrict aux_array2,
    COMPLEX *restrict reverse_shift_matrix );

// This function is implemented in build_next_eigenframe.c
void build_next_eigenframe(
    const REAL *restrict displacements,
    state_struct *restrict state );

// This function is implemented in compute_displacements_and_build_next_eigenframe.c
void compute_displacements_and_build_next_eigenframe(
    state_struct *restrict state,
    REAL *restrict displacements );

// This function is implemented in get_eigenframe.c
void get_eigenframe(
    state_struct *restrict state,
    REAL *restrict eigenframe );

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

static inline
int round_towards_zero(const REAL x) {
  /*
   *  Rounds a number to the nearest integer towards zero.
   *
   *  Inputs
   *  ------
   *    x : Number to be rounded.
   *
   *  Returns
   *  -------
   *    y : Number rounded towards zero.
   */
  if( x > 0.0 )
    return FLOOR(x);
  else
    return CEIL(x);
}

//********************************************
