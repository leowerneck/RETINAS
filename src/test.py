import os
from retina import retina
from numpy import zeros, uint16
from utils import center_array_max_return_displacements

libpath         = os.path.join("..", "lib", "libretina_c.so")
N_horizontal    = 4
N_vertical      = 4
upsample_factor = 100
time_constant   = 10
r = retina(libpath, N_horizontal, N_vertical, upsample_factor, time_constant)

a = zeros((N_horizontal, N_vertical), dtype=uint16)
for i in range(N_horizontal):
    for j in range(N_vertical):
        a[i][j] = i+j

displacements = r.compute_displacements_wrt_ref_image(a)
print(displacements)
displacements = r.compute_displacements_wrt_ref_image(a)
print(displacements)

b, _, _ = center_array_max_return_displacements(a)
displacements = r.compute_displacements_wrt_ref_image(b)
print(displacements)

r.finalize()
