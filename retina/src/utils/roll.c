#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void roll_2d(
    const int m,
    const int n,
    const int h_roll,
    const int v_roll,
    uint16_t *restrict x ) {
  /*
   *  Shifts data in the horizontal and vertical directions.
   *
   *  Inputs
   *  ------
   *    m      : Size of horizontal dimension.
   *    n      : Size of vertical dimension.
   *    h_roll : How many points to roll in horizontal direction.
   *    v_roll : How many points to roll in vertical direction.
   *    x      : Input and output array.
   *
   *  Returns
   *  -------
   *    Nothing.
   *
   *  Example
   *  -------
   *  Consider the following input array:
   *
   *        /  1  2  3  4 \
   *    x = |  5  6  7  8 | .
   *        |  9 10 11 12 |
   *        \ 13 14 15 16 /
   *
   *  We want to roll it one to the right horizontally and
   *  one to the bottom vertically, we would call:
   *
   *  roll_2d(3, 3, 1, 1, x);
   *
   *  and the output array would be
   *
   *        / 16 13 14 15 \
   *    x = |  4  1  2  3 | .
   *        |  8  5  6  7 |
   *        \ 12  9 10 11 /
   */

  // Step 1: Allocate memory for an auxiliary array
  const size_t memsize = sizeof(uint16_t)*m*n;
  uint16_t *y = (uint16_t *)malloc(memsize);

  // Step 2: Copy input array to auxiliary array
  memcpy(y, x, memsize);

  // Step 3: Set auxiliary variables
  const int hh = h_roll%m;
  const int vv = v_roll%n;
  const int mh = m - hh;
  const int nv = n - vv;

  // Step 4: Loop over all points and update x
  // Step 4.1: This first set of loops deals only with points for which j<vv
  for(int j=0;j<vv;j++) {
    const int j_new = j + nv;
    for(int i=0;i<hh;i++) {
      const int i_new   = i + mh;
      const int old_idx = i + m*j;
      const int new_idx = i_new + m*j_new;
      x[old_idx] = y[new_idx];
    }
    for(int i=hh;i<m;i++) {
      const int i_new   = i - hh;
      const int old_idx = i + m*j;
      const int new_idx = i_new + m*j_new;
      x[old_idx] = y[new_idx];
    }
  }

  // Step 4.2: This second set of loops deals only with points for which j>=vv
  for(int j=vv;j<n;j++) {
    const int j_new = j - vv;
    for(int i=0;i<hh;i++) {
      const int i_new   = i + mh;
      const int old_idx = i + m*j;
      const int new_idx = i_new + m*j_new;
      x[old_idx] = y[new_idx];
    }
    for(int i=h_roll;i<m;i++) {
      const int i_new   = i - hh;
      const int old_idx = i + m*j;
      const int new_idx = i_new + m*j_new;
      x[old_idx] = y[new_idx];
    }
  }

  // Step 5: Free allocated memory
  free(y);
}
