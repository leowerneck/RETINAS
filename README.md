# The `RETINAS` Code

The Real time Image Analysis Software (`RETINAS`) is a open-source library
providing `CUDA`, `C`, and `Python` routines for both real-time and offline
sub-pixel registration of input images.

## Installation

`RETINAS` uses the [Meson](https://mesonbuild.com/) build system and defaults to
a [Ninja](https://ninja-build.org/) backend. We recommend using a virtual
environment when using these as that takes care of potential permission issues.
Here's an example of how to set it up:

```shell
python3 -m venv py3
source py3/bin/activate
pip install -U pip
pip install meson ninja
```

### System-wide Installation (default)

As an example, suppose you wish to install both the `C` and `CUDA` versions of
`RETINAS` on your system using `gcc` as the `C` compiler and `nvcc-18` as the
`CUDA compiler`. To do so, run the following commands:

```shell
CC=gcc NVCC=nvcc-18 ./configure --with-cuda=yes
make
make install
```

Note that the `C` version of the library is compiled by default. Note also that
the compilation is quite verbose, but most of the output can be suppressed by
invoking `configure` with the `-s` (or `--silent-compile`) flag.

### Local Installation

Now let us install only the `CUDA` version of `RETINAS` on a path other than the
default (`/usr/local` on `Linux` and `MacOS`), say the base directory of this
repository. We will use the default `CUDA` compiler and suppress most of the
compilation by passing the flag `-s` to configure. Here is the final command:

```shell
./configure -s --prefix=`pwd` --with-c=no --with-cuda=yes
make
make install
```

### Uninstalling

To uninstall `RETINAS`, simply run

```shell
make uninstall
```

Note, however, that this operation requires knowledge of the `--prefix` path. If
this value has changed since the library was installed (e.g., you reconfigured
the build system) then you will need to run the following:

```shell
./configure --prefix=<installation_path>
make uninstall
```

### Dependencies

The following are a list of dependency for different implementations of
`RETINAS`:

* `C` version
  * A `C` compiler (such as `gcc`, `icc`, `icx`, `clang`, ...) with support for `C99`
  * [FFTW3](http://fftw.org/)
  * [BLAS](https://netlib.org/blas/)

* `CUDA` version
  * A `CUDA` compiler (`nvcc`)
  * [cuBLAS](https://developer.nvidia.com/cublas)
  * [cuFFT](https://developer.nvidia.com/cufft)

* `Python` version
  * [Python 3](https://www.python.org/)
  * [NumPy](https://numpy.org/)
  * [SciPy](https://scipy.org/)

## Synthetic data test

To run the synthetic data test on `Linux` you can run:

```shell
./configure --prefix=`pwd` --with-cuda=yes
make
make install
cd retinas
python synthetic_data_test.py
```

It should take a few minutes for the test to complete. Some progress information will be printed periodically. Once the test finishes, the following files will be generated:

1. `diagnostics.txt`: contains a nearly formatted output for all three implementations of the code (`C`, `CUDA`, and `Python`).
2. `out/displacements_w8_o5.76.txt`: contains the analytic displacements
3. `out/results.txt`: contains the numerical displacements for all three implementations

## Offline analysis (In progress)

Here we provide an example of how to perform "offline" analysis. This is the case where we keep the reference image fixed and only update it after a certain number of images have been processed. Currently this algorithm is only available in the `Python` implementation of `RETINAS`.

```Python
from pyretinas import Pyretinas

# Initialize the Pyretinas object
rpy = Pyretinas(N_horizontal, N_vertical, upsample_factor, time_constant, shot_noise=shot_noise, offset=offset)

# Loop over as many images as we want, computing the displacements
for i in range(N_images):
    brightness    = rpy.preprocess_new_image_and_compute_brightness(im)
    # This should keep the reference frame fixed, but will add the FFT of the
    # new image to an accumulator (i.e., image_sum_freq += new_image_freq)
    displacements = rpy.compute_displacements_and_add_new_image_to_sum()
    #
    # <do extra stuff>
    #

# Once N_images are read, for example, we update the reference image. This will
# do three things:
#
# 1. Set ref_frame_freq = image_sum_freq/N_images (note that image_sum_freq
#                                                  includes ref_frame_freq)
#
# 2. Reset the image sum to the current reference frame, i.e.,
#        image_sum_freq = ref_frame_freq
#
# 3. Reset the image counter to one.
rpy.update_reference_image_from_image_sum()

rpy.finalize()
```
