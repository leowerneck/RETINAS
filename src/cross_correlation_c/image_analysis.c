#include "image_analysis.h"

int dump_eigenframe( Cstate_struct *restrict Cstate ) {
  const int Nh = Cstate->N_horizontal;
  const int Nv = Cstate->N_vertical;
  FFTW_EXECUTE_DFT(Cstate->ifft2_plan, Cstate->eigenframe_freq_domain, Cstate->new_image_time_domain);
  FILE *fp = fopen("eigenframe_cross_correlation_C.txt", "w");
  for(int i=0;i<Nh;i++) {
    for(int j=0;j<Nv;j++) {
      const COMPLEX z = Cstate->new_image_time_domain[i+Nh*j];
      // const COMPLEX w = Cstate->eigenframe_freq_domain[i+Nh*j];
      fprintf(fp, "%d %d %.15e %.15e\n", i, j, CREAL(z)/(Nh*Nv), CIMAG(z)/(Nh*Nv));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  return 0;
}

int set_zeroth_eigenframe( Cstate_struct *restrict Cstate ) {

  // Step 1: Compute FFT of the new_image_time_domain and
  //         store it as the zeroth eigenframe.
  FFTW_EXECUTE_DFT(Cstate->fft2_plan,Cstate->new_image_time_domain,Cstate->eigenframe_freq_domain);

  // Step 2: All done!
  return 0;

}

// This function is used to compute a reverse shift.
void compute_reverse_shift_2D(const int N_horizontal,
                              const int N_vertical,
                              const REAL *restrict displacements,
                              COMPLEX *restrict aux_array1,
                              COMPLEX *restrict aux_array2,
                              COMPLEX *restrict displacement_matrix) {

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
      displacement_matrix[i+N_horizontal*j] = shift_horizontal*shift_vertical;
    }
  }
  displacement_matrix[Nhover2+N_horizontal*Nvover2] = CREAL(displacement_matrix[Nhover2+N_horizontal*Nvover2]);
}

int cross_correlate_and_build_next_eigenframe(Cstate_struct *restrict Cstate,
                                              REAL *restrict displacement) {

  // Step 1: Compute the FFT of the new image
  FFTW_EXECUTE_DFT(Cstate->fft2_plan,Cstate->new_image_time_domain,Cstate->new_image_freq_domain);

  // Step 2: Compute image product target * src^{*}
  const int N_horizontal = Cstate->N_horizontal;
  const int N_vertical   = Cstate->N_vertical;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      Cstate->aux_array1[idx] = Cstate->new_image_freq_domain[idx]*CONJ(Cstate->eigenframe_freq_domain[idx]);
    }
  }

  // Step 3: Compute the cross correlation
  // Note: aux_array1 stores the image product and
  //       aux_array2 stores the cross correlation
  FFTW_EXECUTE_DFT(Cstate->ifft2_plan,Cstate->aux_array1,Cstate->aux_array2);

  // FILE *fp = fopen("first_frame.txt","w");
  // for(int j=0;j<Cstate->N_vertical;j++) {
  //   for(int i=0;i<Cstate->N_horizontal;i++) {
  //     const int index = i + Cstate->N_horizontal*j;
  //     const COMPLEX z1 = Cstate->new_image_time_domain[index];
  //     const COMPLEX z2 = Cstate->new_image_freq_domain[index];
  //     const COMPLEX z3 = Cstate->eigenframe_freq_domain[index];
  //     const COMPLEX z4 = Cstate->aux_array1[index];
  //     const COMPLEX z5 = Cstate->aux_array2[index];
  //     fprintf(fp,"%d %d %e %e %e %e %e %e %e %e %e %e\n",i,j,
  //             CREAL(z1),CIMAG(z1),
  //             CREAL(z2),CIMAG(z2),
  //             CREAL(z3),CIMAG(z3),
  //             CREAL(z4),CIMAG(z4),
  //             CREAL(z5),CIMAG(z5));
  //   }
  //   fprintf(fp,"\n");
  // }
  // fclose(fp);

  // Step 4: Full-pixel estimate of the cross correlation maxima
  int i_max=0,j_max=0;
  REAL cc_max = -1.0;
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i + N_horizontal*j;
      const REAL cc = CABS(Cstate->aux_array2[idx]);
      if( cc > cc_max ) {
        cc_max = cc;
        i_max  = i;
        j_max  = j;
      }
    }
  }
  displacement[0] = (REAL)i_max;
  displacement[1] = (REAL)j_max;

  // Step 5: Compute midpoint
  const int midpoint[2] = {N_horizontal/2,N_vertical/2};

  // Step 6: If the displacement is larger than the midpoint,
  //         then subtract N_horizontal or N_vertical, accordingly
  if( displacement[0] > midpoint[0] ) displacement[0] -= N_horizontal;
  if( displacement[1] > midpoint[1] ) displacement[1] -= N_vertical;

  // Step 7: Upsampled the image around the maxima and
  //         find new maxima with sub-pixel precision
  // Note: aux_array1 stores the image product, while
  //       aux_array2 and aux_array3 can be used for
  //       any purpose
  const REAL upsample_factor = Cstate->upsample_factor;
  if( (int)(upsample_factor+0.5) > 1 ) {
    get_subpixel_displacement_by_upsampling(N_horizontal,N_vertical,upsample_factor,Cstate->ifft2_plan,
                                            Cstate->aux_array1,Cstate->aux_array2,Cstate->aux_array3,
                                            displacement);
  }

  // Step 8: Loop over all images in the dataset, reverse shift
  //         them (in Fourier space) and add them to the eigenframe
  compute_reverse_shift_2D(N_horizontal,N_vertical,displacement,
                           Cstate->aux_array1,Cstate->aux_array2,Cstate->aux_array3);

  // Step 9: Now shift the image and add it to the eigenframe
  for(int j=0;j<N_vertical;j++) {
    for(int i=0;i<N_horizontal;i++) {
      const int idx = i+N_horizontal*j;
      Cstate->eigenframe_freq_domain[idx] = Cstate->A0*(Cstate->new_image_freq_domain[idx]*Cstate->aux_array3[idx]) + Cstate->B1*Cstate->eigenframe_freq_domain[idx];
    }
  }

  return 0;
}
