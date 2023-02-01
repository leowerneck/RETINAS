#include "image_analysis.h"

// C = (A.B)^{T} = B^{T}.A^{T}, where:
// A is m x p, A^{T} is p x m
// B is p x n, B^{T} is n x p
// C is n x m
extern "C" __host__
void complex_matrix_multiply_tt(cublasHandle_t cublasHandle,
                                const int m,
                                const int p,
                                const int n,
                                const COMPLEX *restrict A,
                                const COMPLEX *restrict B,
                                COMPLEX *restrict C) {

  const COMPLEX alpha = MAKE_COMPLEX(1.0,0.0);
  const COMPLEX beta  = MAKE_COMPLEX(0.0,0.0);
  CUBLASCGEMM(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_T,
              n,m,p,&alpha,B,p,A,m,&beta,C,m);

}

// C = A.B, where:
// A is m x p
// B is p x n
// C is m x n
extern "C" __host__
void complex_matrix_multiply(cublasHandle_t cublasHandle,
                             const int m,
                             const int p,
                             const int n,
                             const COMPLEX *restrict A,
                             const COMPLEX *restrict B,
                             COMPLEX *restrict C) {

  const COMPLEX alpha = MAKE_COMPLEX(1.0,0.0);
  const COMPLEX beta  = MAKE_COMPLEX(0.0,0.0);
  CUBLASCGEMM(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,
              m,n,p,&alpha,A,m,B,p,&beta,C,m);

}