#include <complex.h>
#include <immintrin.h>

void cmulf_conjf(
    float complex const *restrict z1,
    float complex const *restrict z2,
    float complex *restrict z3 ) {
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

void cmulf_conjf_simd(
    float complex const *restrict z1,
    float complex const *restrict z2,
    float complex *restrict z3 ) {
  /*
   *  Using SIMD intrinsics, this function computes the products:
   *
   *    z3[0] = z1[0].z2[0]^{*}
   *    z3[1] = z1[1].z2[1]^{*}
   *    z3[2] = z1[2].z2[2]^{*}
   *    z3[3] = z1[3].z2[3]^{*}
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
   *      z1[0] = a1 + ib1 ,
   *      z1[1] = c1 + id1 ,
   *      z1[2] = e1 + if1 ,
   *      z1[3] = g1 + ih1 ,
   *      z2[0] = a2 + ib2 ,
   *      z2[1] = c2 + id2 ,
   *      z2[2] = e2 + if2 ,
   *      z2[3] = g2 + ih2 ,
   *
   *    In memory, these entries are stored as follows:
   *
   *      z1 -> a1 b1 c1 d1 e1 f1 g1 h1 ,
   *      z2 -> a2 b2 c2 d2 e2 f2 g2 h2 .
   *
   *    The products we are interested in computing are:
   *
   *    z3[0] = (a1 + ib1)(a2 - ib2) = (a1a2 + b1b2) + i(b1a2 - a1b2) ,
   *    z3[1] = (c1 + id1)(c2 - id2) = (c1c2 + d1d2) + i(d1c2 - c1d2) ,
   *    z3[2] = (e1 + if1)(e2 - if2) = (e1e2 + f1f2) + i(f1e2 - e1f2) ,
   *    z3[3] = (g1 + ih1)(g2 - ih2) = (g1g2 + h1h2) + i(h1g2 - g1h2) ,
   *
   *    The code below uses a series of SIMD intrinsics to compute these four
   *    numbers efficiently.
   */

  __m256 m1p1m1p1              = _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
  __m256 abcd1                 = _mm256_load_ps((float *)&z1[0]);
  __m256 abcd2                 = _mm256_load_ps((float *)&z2[0]);
  __m256 badc2                 = _mm256_permute_ps(abcd2, 0b10110001);
  __m256 aa_bb_cc_dd           = _mm256_mul_ps(abcd1, abcd2);
  __m256 ab_ba_cd_dc           = _mm256_mul_ps(abcd1, badc2);
  __m256 mab_pba_mcd_pdc       = _mm256_mul_ps(ab_ba_cd_dc, m1p1m1p1);
  __m256 aabb_ccdd_bamab_dcmcd = _mm256_hadd_ps(aa_bb_cc_dd, mab_pba_mcd_pdc);
  __m256 aabb_bamab_ccdd_dcmcd = _mm256_permute_ps(aabb_ccdd_bamab_dcmcd, 0b11011000);
  _mm256_store_ps((float *)&z3[0], aabb_bamab_ccdd_dcmcd);
}
