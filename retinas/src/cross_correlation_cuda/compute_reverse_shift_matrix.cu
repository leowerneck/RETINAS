#include "retinas.h"

__global__
static void compute_shift_1d_gpu(
    const int N,
    const REAL displacement,
    COMPLEX *restrict shift ) {

  // Step 1: Get thread index and stride
  int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Step 2: Compute useful quantities
  const int Nover2    = N/2;
  const REAL oneoverN = 1.0/N;
  const COMPLEX aux   = MAKE_COMPLEX(0.0, 2*M_PI*displacement);

  // Step 3: Compute the shift
  for(int i=tid;i<N;i+=stride) {
    shift[i       ] = CEXP(CMUL(aux, MAKE_COMPLEX(i * oneoverN,0.0)));
    shift[i+Nover2] = CEXP(CMUL(aux, MAKE_COMPLEX((i+Nover2)*oneoverN-1,0.0)));
  }
  if(tid==0) shift[Nover2] = MAKE_COMPLEX(CREAL(shift[Nover2]),0.0);

}

// This function is used to compute a reverse shift.
__global__
static void compute_reverse_shift_2d_gpu(
    const int N_horizontal,
    const int N_vertical,
    const COMPLEX *restrict horizontal_shifts,
    const COMPLEX *restrict vertical_shifts,
    COMPLEX *restrict reverse_shift ) {

  // Step 1: Set the thread indices
  int tidx    = threadIdx.x + blockIdx.x*blockDim.x;
  int tidy    = threadIdx.y + blockIdx.y*blockDim.y;
  int stridex = blockDim.x*gridDim.x;
  int stridey = blockDim.y*gridDim.y;

  // Step 2: Compute the reverse shift
  for(int i_y=tidy;i_y<N_vertical;i_y+=stridey) {
    const COMPLEX vshift = vertical_shifts[i_y];
    for(int i_x=tidx;i_x<N_horizontal;i_x+=stridex) {
      const COMPLEX hshift = horizontal_shifts[i_x];
      reverse_shift[i_x + N_horizontal*i_y] = CMUL(hshift, vshift);
    }
  }

}

extern "C" __host__
void compute_reverse_shift_matrix(
    const int N_horizontal,
    const int N_vertical,
    const REAL *restrict displacements,
    COMPLEX *restrict horizontal_shifts,
    COMPLEX *restrict vertical_shifts,
    COMPLEX *restrict shift_matrix ) {

  // Step 1: Compute the horizontal shift
  const int Nho2 = N_horizontal/2;
  compute_shift_1d_gpu<<<MIN(Nho2,512),MIN(Nho2,512)>>>(N_horizontal, displacements[0], horizontal_shifts);

  // Step 2: Compute the vertical shift
  const int Nvo2 = N_vertical/2;
  compute_shift_1d_gpu<<<MIN(Nvo2,512),MIN(Nvo2,512)>>>(N_vertical, displacements[1], vertical_shifts);

  // Step 3: Compute the reverse shift 2D
  compute_reverse_shift_2d_gpu<<<dim3(32,16),dim3(32,16)>>>(N_horizontal, N_vertical,
                                                            horizontal_shifts, vertical_shifts,
                                                            shift_matrix );
}
