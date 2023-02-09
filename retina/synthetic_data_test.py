from os.path import join as pjoin
from time import time
from retina import retina
from numpy import uint16, fromfile
from utils import generate_synthetic_image_data_set

def rel_err(a, b):
    if a+b == 0.0:
        return 0.0
    if a != 0.0:
        return fabs(1-b/a)
    return fabs(1-a/b)

def abs_err(a, b):
    return fabs(a-b)

libpath         = pjoin("..", "lib", "libretina_cuda.so")
N_horizontal    = 256
N_vertical      = 128
upsample_factor = 100
time_constant   = 10
offset          = 10
A               = 10700
w               = 16
spread_factor   = 0.95
outdir          = "out"
N_images        = 1000
imdir = generate_synthetic_image_data_set(outdir, N_images, N_horizontal, N_vertical,
                                          A=A, w=w, offset=offset, spread_factor=0.95)

r = retina(libpath, N_horizontal, N_vertical, upsample_factor, time_constant, precision="double")
print("(RETINA) Beginning image processing")
start = time()
with open(pjoin(outdir, "results.txt"), "w") as file:
    im = fromfile(pjoin(imdir, "image_01.bin"), dtype=uint16).reshape(N_vertical,N_horizontal)
    displacements = r.compute_displacements_wrt_ref_image_and_build_next_eigenframe(im)
    file.write(f"{displacements[0]:.15e} {displacements[1]:.15e}\n")
    for i in range(1,N_images+1):
        im = fromfile(pjoin(imdir, f"image_{i+1:02d}.bin"), dtype=uint16).reshape(N_vertical,N_horizontal)
        displacements = r.compute_displacements_wrt_ref_image_and_build_next_eigenframe(im)
        file.write(f"{displacements[0]:.15e} {displacements[1]:.15e}\n")
        if not i%(N_images/5):
            print(f"(RETINA) Finished processing image {i:05d} of {N_images:05d}")

r.finalize()
end = time()
print(f"(RETINA) Finished processing {N_images} images of size {N_horizontal} x {N_vertical} in {end-start} seconds")
