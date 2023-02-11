#ifndef IMAGE_ANALYSIS_H_
#define IMAGE_ANALYSIS_H_

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

// CUDA complex type
#include <cuComplex.h>

// cuFFT library
#include <cufft.h>

// cuBLAS library
#include <cublas_v2.h>

// Precision macros
#define SINGLE (0)
#define DOUBLE (1)
#define PRECISION DOUBLE

// Useful macros
#if PRECISION == SINGLE
#  define REAL float
#  define ROUND roundf
#  define FLOOR floorf
#  define CEIL ceilf
#  define COMPLEX cuFloatComplex
#  define MAKE_COMPLEX make_cuFloatComplex
#  define CREAL cuCrealf
#  define CIMAG cuCimagf
#  define ABS fabsf
#  define CABS cuCabsf
#  define CONJ cuConjf
#  define SIN sinf
#  define COS cosf
#  define EXP expf
#  define CMUL cuCmulf
#  define FFT_PLAN cufftHandle
#  define FFT_PLAN_DFT_2D cufftPlan2d
#  define FFT_EXECUTE_DFT cufftExecC2C
#  define FFT_DESTROY_PLAN cufftDestroy
#  define FFT_C2C CUFFT_C2C
#  define CUBLASCGEMM cublasCgemm
#  define CUBLASIAMAX cublasIsamax
#  define CUBLASIAMIN cublasIsamin
#  define CUBLASASUM cublasSasum
#elif PRECISION == DOUBLE
#  define REAL double
#  define ROUND round
#  define FLOOR floor
#  define CEIL ceil
#  define COMPLEX cuDoubleComplex
#  define MAKE_COMPLEX make_cuDoubleComplex
#  define CREAL cuCreal
#  define CIMAG cuCimag
#  define CABS cuCabs
#  define ABS fabs
#  define CONJ cuConj
#  define SIN sin
#  define COS cos
#  define EXP exp
#  define CMUL cuCmul
#  define FFT_PLAN cufftHandle
#  define FFT_PLAN_DFT_2D cufftPlan2d
#  define FFT_EXECUTE_DFT cufftExecZ2Z
#  define FFT_DESTROY_PLAN cufftDestroy
#  define FFT_C2C CUFFT_Z2Z
#  define CUBLASCGEMM cublasZgemm
#  define CUBLASIAMAX cublasIdamax
#  define CUBLASIAMIN cublasIdamin
#  define CUBLASASUM cublasDasum
#else
#  error "Unknown precision"
#endif

// Image analysis parameter struct
typedef struct state_struct {
  bool shot_noise_method;
  int N_horizontal, N_vertical;
  REAL upsample_factor, A0, B1, shift;
  uint16_t *aux_array_int;            // GPU (device)
  REAL *aux_array_real;               // GPU (device)
  COMPLEX *aux_array1;                // GPU (device)
  COMPLEX *aux_array2;                // GPU (device)
  COMPLEX *aux_array3;                // GPU (device)
  COMPLEX *new_image_time;            // GPU (device)
  COMPLEX *new_image_freq;            // GPU (device)
  COMPLEX *eigenframe_freq;           // GPU (device)
  COMPLEX *reciprocal_new_image_time; // GPU (device)

  // No memory is allocated for these, but they are
  // useful and more descriptive than "aux_array"
  COMPLEX *eigenframe_time;
  COMPLEX *image_product;
  COMPLEX *cross_correlation;
  COMPLEX *upsampled_image;
  COMPLEX *horizontal_kernel;
  COMPLEX *vertical_kernel;
  COMPLEX *partial_product;
  COMPLEX *horizontal_shifts;
  COMPLEX *vertical_shifts;
  COMPLEX *shift_matrix;

  FFT_PLAN fft2_plan;
  cublasHandle_t cublasHandle;
} state_struct;

#include "function_prototypes.h"

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
#endif // IMAGE_ANALYSIS_H_
