#include "retinas.h"

/*
 *  Function: compute_reverse_shift_matrix
 *  Author  : Leo Werneck
 *
 *  Compute the reverse shift matrix.
 *
 *  Arguments
 *  ---------
 *    N_horizontal : in
 *      Number of horizontal points in the images.
 *
 *    N_vertical : in
 *      Number of vertical points in the images.
 *
 *    displacements : in
 *      Horizontal and vertical displacements.
 *
 *    aux_array1 : in/out
 *      Empty auxiliary array.
 *
 *    aux_array2 : in/out
 *      Empty auxiliary array.
 *
 *    reverse_shift_matrix : out
 *      The reverse shift matrix.
 *
 *  Returns
 *  -------
 *    Nothing.
 */
void compute_reverse_shift_matrix(
    const int N_horizontal,
    const int N_vertical,
    const REAL *restrict displacements,
    COMPLEX *restrict aux_array1,
    COMPLEX *restrict aux_array2,
    COMPLEX *restrict reverse_shift_matrix ) {

  // Step 1: Set useful quantities
  const int Nhover2    = N_horizontal/2;
  const int Nvover2    = N_vertical/2;
  const COMPLEX Ipi2   = 0.0 + I * M_PI * 2.0;
  const COMPLEX aux_h  = Ipi2 * displacements[0]; // Horizontal
  const COMPLEX aux_v  = Ipi2 * displacements[1]; // Vertical
  const REAL oneoverNh = 1.0/((REAL)N_horizontal);
  const REAL oneoverNv = 1.0/((REAL)N_vertical);

  // Step 2: Pre-compute the displacement coefficients
  //         in the horizontal and vertical directions
  for(int i=0;i<Nhover2;i++) {
    aux_array1[i        ] = CEXP(aux_h*i*oneoverNh);
    aux_array1[i+Nhover2] = CEXP(aux_h*((i+Nhover2)*oneoverNh-1));
  }
  for(int j=0;j<Nvover2;j++) {
    aux_array2[j        ] = CEXP(aux_v*j*oneoverNv);
    aux_array2[j+Nvover2] = CEXP(aux_v*((j+Nvover2)*oneoverNv-1));
  }

  // Step 3: Compute the displacement matrix
  for(int j=0;j<N_vertical;j++) {
    const COMPLEX shift_vertical = aux_array2[j];
    for(int i=0;i<N_horizontal;i++) {
      const COMPLEX shift_horizontal = aux_array1[i];
      reverse_shift_matrix[i+N_horizontal*j] = shift_horizontal*shift_vertical;
    }
  }
  reverse_shift_matrix[Nhover2+N_horizontal*Nvover2] = CREAL(reverse_shift_matrix[Nhover2+N_horizontal*Nvover2]);
}
