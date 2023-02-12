#include <complex.h>
#include <immintrin.h>

void cmul_conj(
    double complex const *restrict z1,
    double complex const *restrict z2,
    double complex *restrict z3 ) {
  /*
   *  Compute the product
   *
   *   z3 = z1.z2^{*} ,
   *
   *  where * denotes complex conjugation.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      z1 : First  vector of complex numbers.
   *      z2 : Second vector of complex numbers.
   *
   *    Output
   *    ------
   *      z3 : Stores the results.
   */

  *z3 = (*z1)*conjf(*z2);
}

static inline
void cmul_conj_simd(
    double complex const *restrict z1,
    double complex const *restrict z2,
    double complex *restrict z3 ) {
  /*
   *  Using SIMD intrinsics, this function computes the products:
   *
   *    z3[0] = z1[0].z2[0]^{*}
   *    z3[1] = z1[1].z2[1]^{*}
   *
   *  where * denotes complex conjugation, using SIMD intrinsics.
   *
   *  Arguments
   *  ---------
   *    Inputs
   *    ------
   *      z1 : First  vector of complex numbers.
   *      z2 : Second vector of complex numbers.
   *
   *    Output
   *    ------
   *      z3 : Stores the results.
   *
   *  Mathematical details and memory layout
   *  --------------------------------------
   *    Let us consider the following notation:
   *
   *      z1[0] = a + ib ,
   *      z1[1] = c + id ,
   *      z2[0] = e + if ,
   *      z3[0] = g + ih .
   *
   *    In memory, these entries are stored as follows:
   *
   *      z1 -> a b c d ,
   *      z2 -> e f g h .
   *
   *    The products we are interested in computing are:
   *
   *    z3[0] = z1[0].z2[0]^{*} = (a + ib)(e - if) = (ae + bf) + i(be - af) ,
   *    z3[1] = z1[1].z2[1]^{*} = (c + id)(g - ih) = (cg + dh) + i(dg - ch) .
   *
   *    The code uses a series of SIMD intrinsics to compute these two numbers
   *    efficiently.
   */

  __m256d m1p1m1p1        = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
  __m256d abcd            = _mm256_load_pd((double *)z1);
  __m256d efgh            = _mm256_load_pd((double *)z2);
  __m256d fehg            = _mm256_permute_pd(efgh, 0b0101);
  __m256d ae_bf_cg_dh     = _mm256_mul_pd(abcd, efgh);
  __m256d af_be_ch_dg     = _mm256_mul_pd(abcd, fehg);
  __m256d maf_pbe_mch_pdg = _mm256_mul_pd(af_be_ch_dg,m1p1m1p1);
  _mm256_store_pd((double *)z3, _mm256_hadd_pd(ae_bf_cg_dh, maf_pbe_mch_pdg));
}
