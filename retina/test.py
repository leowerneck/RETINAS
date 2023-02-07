import os
import sys
import time
from retina import retina
from numpy import zeros, uint16, fabs, sum
from numpy.random import random
from utils import Poisson_image

def rel_err(a, b):
    if a+b == 0.0:
        return 0.0
    if a != 0.0:
        return fabs(1-b/a)
    return fabs(1-a/b)

def abs_err(a, b):
    return fabs(a-b)

libpath         = os.path.join("..", "lib", "libretina_cuda.so")
N_horizontal    = 256
N_vertical      = 128
upsample_factor = 100
time_constant   = 10
spread_factor   = 0.95

r = retina(libpath, N_horizontal, N_vertical, upsample_factor, time_constant, precision="double")

start = time.time()
N_images = 1000000
with open("displacements.txt", "w") as file:
    dh = 0
    dv = 0
    im = Poisson_image((N_horizontal,N_vertical), c=(dh,dv))
    displacements = r.compute_displacements_wrt_ref_image_and_build_next_eigenframe(im)
    file.write("%.15e %.15e %.15e %.15e\n"%(
        dh, dv, displacements[0], displacements[1]))
    for i in range(1,N_images+1):
        dh = spread_factor*(random()-0.5)*N_horizontal
        dv = spread_factor*(random()-0.5)*N_vertical
        im = Poisson_image((N_horizontal,N_vertical), c=(dh,dv))
        displacements = r.compute_displacements_wrt_ref_image_and_build_next_eigenframe(im)
        dh_abs_err = abs_err(displacements[0], dh)
        dv_abs_err = abs_err(displacements[1], dv)
        dh_rel_err = rel_err(displacements[0], dh)
        dv_rel_err = rel_err(displacements[1], dv)
        file.write("%.15e %.15e %.15e %.15e\n"%(
            dh, dv, displacements[0], displacements[1]))
        print(f"(RETINA) Finished processing image {i:05d} of {N_images:05d}")

r.finalize()
end = time.time()
print(f"Finished processing {N_images} images of size {N_horizontal} x {N_vertical} in {end-start} seconds")
