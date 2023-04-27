from os.path import join as pjoin
from time import time
from retinas import retinas
from numpy import uint16, fromfile
from utils import generate_synthetic_image_data_set
from pyretinas import Pyretinas

def rel_err(a, b):
    if a+b == 0.0:
        return 0.0
    if a != 0.0:
        return fabs(1-b/a)
    return fabs(1-a/b)

def abs_err(a, b):
    return fabs(a-b)

libpath_c       = pjoin("..", "lib", "x86_64-linux-gnu", "libretinas.so")
libpath_cuda    = pjoin("..", "lib", "x86_64-linux-gnu", "libretinas_cuda.so")
N_horizontal    = 256
N_vertical      = 128
upsample_factor = 256
time_constant   = 10
offset          = 5.76
shot_noise      = False
A               = 10700
w               = 8
spread_factor   = 0.95
outdir          = "out"
N_images        = 1000

print("(RETINAS) This routine will test all three implementations (C, CUDA, Python).")
print("(RETINAS) Neatly formatted diagnostics can be found in the file diagnostics.txt.")
print(f"(RETINAS) File {pjoin(outdir, 'results.txt')} contains analytic and numerical results")

imdir = generate_synthetic_image_data_set(outdir, N_images,
                                          (N_horizontal,N_vertical),
                                          A=A, w=w, offset=offset,
                                          spread_factor=spread_factor)

rc    = retinas(libpath_c   , N_horizontal, N_vertical, upsample_factor, time_constant, precision="double", shot_noise=shot_noise, offset=offset)
rcuda = retinas(libpath_cuda, N_horizontal, N_vertical, upsample_factor, time_constant, precision="double", shot_noise=shot_noise, offset=offset)
rpy   = Pyretinas(            N_horizontal, N_vertical, upsample_factor, time_constant, shot_noise=shot_noise, offset=offset)

print("(RETINAS) Beginning image processing")

start = time()
n = len("Implementation")
div = ("%*s . %22s . %22s"%(n, " ", " ", " ")).replace(" ", "-")
finfo = open("diagnostics.txt", "w")
finfo.write(div+"\n")
finfo.write("%*s | %*s%*s | %*s%*s\n"%(n, "Implementation",
                                       int((22+len("Hor. Displ."))//2),
                                       "Hor.Displ.",
                                       int(22-(22+len("Hor. Displ."))//2),
                                       " ",
                                       int((22+len("Vert. Displ."))//2),
                                       "Vert. Displ.",
                                       int(22-(22+len("Vert. Displ."))//2),
                                       " "))
finfo.write(div+"\n")
with open(pjoin(outdir, "results.txt"), "w") as f:
    for i in range(N_images+1):
        im    = fromfile(pjoin(imdir, f"image_{i+1:02d}.bin"), dtype=uint16).reshape(N_vertical,N_horizontal)
        bc    = rc.preprocess_new_image_and_compute_brightness(im)
        bcuda = rcuda.preprocess_new_image_and_compute_brightness(im)
        bpy   = rpy.preprocess_new_image_and_compute_brightness(im)
        dc    = rc.compute_displacements_and_update_ref_image()
        dcuda = rcuda.compute_displacements_and_update_ref_image()
        dpy   = rpy.compute_displacements_and_update_ref_image()
        finfo.write("%*s%*s | %22.15e | %22.15e\n"%(int(n+len("C"))//2, "C", n-(n+len("C"))//2, " ", dc[0], dc[1]))
        finfo.write("%*s%*s | %22.15e | %22.15e\n"%(int(n+len("CUDA"))//2, "CUDA", n-(n+len("CUDA"))//2, " ", dcuda[0], dcuda[1]))
        finfo.write("%*s%*s | %22.15e | %22.15e\n"%(int(n+len("Python"))//2, "Python", n-(n+len("Python"))//2, " ", dpy[0], dpy[1]))
        finfo.write(div+"\n")
        f.write(f"{dc[0]:.15e} {dc[1]:.15e} {dcuda[0]:.15e} {dcuda[1]:.15e} {dpy[0]:.15e} {dpy[1]:.15e}\n")
        if not i%(N_images/5):
            print(f"(RETINAS) Finished processing image {i:05d} of {N_images:05d}")

rc.finalize()
rcuda.finalize()
rpy.finalize()
end = time()
print(f"(RETINAS) Finished processing {N_images} images of size {N_horizontal} x {N_vertical} in {end-start:.2f} seconds")
