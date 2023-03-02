#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884L
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039L
#endif

static inline double rand_uniform() {
  return ((double)rand())/((double)RAND_MAX);
}

static double gamma_large(const double a) {
  /* Works only if a > 1, and is most efficient if a is large

     This algorithm, reported in Knuth, is attributed to Ahrens.  A
     faster one, we are told, can be found in: J. H. Ahrens and
     U. Dieter, Computing 12 (1974) 223-246.  */
  double sqa, x, y, v;
  sqa = sqrt (2 * a - 1);
  do {
    do {
      y = tan (M_PI * rand_uniform());
      x = sqa * y + a - 1;
    } while (x <= 0);
    v = rand_uniform();
  } while (v > (1 + y * y) * exp ((a - 1) * log (x / (a - 1)) - sqa * y));

  return x;
}

double gamma_int(const unsigned int a) {
  if(a < 12) {
    double prod = 1;
    for (unsigned int i=0;i<a;i++)
      prod *= rand_uniform();

    /* Note: for 12 iterations we are safe against underflow, since
       the smallest positive random number is O(2^-32). This means
       the smallest possible product is 2^(-12*32) = 10^-116 which
       is within the range of double precision. */
    return -log (prod);
  }
  else
    return gamma_large((double) a);
}

unsigned int poisson(double mu) {

  double prod = 1.0;
  unsigned int k = 0;

  while(mu > 10) {
    unsigned int m = mu * (7.0 / 8.0);
    double X = gamma_int(m);

    if (X >= mu)
      return k + gsl_ran_binomial(mu / X, m - 1);
    else {
      k += m;
      mu -= X;
    }
  }

  /* This following method works well when mu is small */
  double emu = exp(-mu);

  do {
    prod *= rand_uniform();
    k++;
  } while (prod > emu);

  return k - 1;
}
