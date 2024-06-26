""" Utility functions for the RETINAS toolkit """

from os import makedirs
from os.path import join as pjoin
from shutil import rmtree
from time import time
from ctypes import c_bool, c_int, POINTER, CDLL
from numpy import unravel_index, argmax, argmin, rint, roll
from numpy import mgrid, sqrt, uint16, pi, multiply, array, exp, outer
from numpy.random import random, poisson
from scipy.special import erf


def setup_library_function(func, params, returns):
    """
    Setup the C functions

    Inputs
    ------
        func : ctypes._FuncPtr
            Pointer to a C function.

        params : list of _ctypes.PyCSimpleType
            List of C types for function parameters.

        returns : _ctypes.PyCSimpleType or None
            Return value of the C function.

    Returns
    -------
        Nothing.

    Raises
    ------
        TypeError : If params have the wrong type.
        TypeError : If any param in params have the wrong type.
        TypeError : If returns have the wrong type.
    """

    # Step 2.a: Check inputs have the correct types
    Ctype = type(c_int)
    Cptrtype = type(POINTER(c_int))
    if not isinstance(params, list):
        raise TypeError(
            f'params must be of type list, but got {type(params)} instead.')
    for param in params:
        if not isinstance(param, Ctype) and not isinstance(param, Cptrtype):
            raise TypeError(
                f'params must be a list of C types, but found {type(param)}.')
    if not isinstance(returns, Ctype) and returns is not None:
        raise TypeError(
            f'returns must be a C type or None, but got {type(returns)} instead.')

    # Step 2.b: Setup the types of function parameters and return value
    func.argtypes = params
    func.restype = returns


def center_array_max_return_displacements(image_array):
    """ Fast recentering of an image array around the maximum pixel """

    shape = image_array.shape
    ind_max = unravel_index(argmax(image_array, axis=None), shape)
    move_0 = int(rint(shape[0]/2 - (ind_max[0]+0.5)+0.1))
    move_1 = int(rint(shape[1]/2 - (ind_max[1]+0.5)+0.1))
    h_0 = -move_1
    v_0 = -move_0
    return roll(image_array, (move_0, move_1), axis=(0, 1)), h_0, v_0


def center_array_min_return_displacements(image_array):
    """ Fast recentering of an image array around the minimum pixel """

    shape = image_array.shape
    ind_min = unravel_index(argmin(image_array, axis=None), shape)
    move_0 = int(rint(shape[0]/2 - (ind_min[0]+0.5)+0.1))
    move_1 = int(rint(shape[1]/2 - (ind_min[1]+0.5)+0.1))
    h_0 = -move_1
    v_0 = -move_0
    return roll(image_array, (move_0, move_1), axis=(0, 1)), h_0, v_0


def Gaussian_image(n, w=1, A=4095, c=(0, 0), offset=0):
    """
    Normal distribution (Gaussian) image. Supports both dark and
    bright background.

    Inputs
    ------
        n : tuple of ints
            Image size.

        w : float
            Gaussian width. Default is 1.

        A : float
            Gaussian amplitude. Default is 4095 = 2**12 - 1.

        c : tuple of floats
            Gaussian center. Default is (0,0).

        offset : float
            The offset simulates background light noise level. Default is 0.

    Returns
    -------
        image : ndarray (dtype=uint16)
            A 2D image of shape (n[0],n[1]).
    """

    if not isinstance(n, tuple) or len(n) != 2:
        raise TypeError("Image size must be a tuple of length 2.")
    if not isinstance(c, tuple) or len(c) != 2:
        raise TypeError("Gaussian center must be a tuple of length 2.")

    # Step 1: Define the vertical and horizontal points of the image
    y, x = mgrid[0:n[1], 0:n[0]]
    x = x - n[0]/2 - c[0]
    y = y - n[1]/2 - c[1]

    # Step 2: Set the image
    w2 = w/sqrt(2)
    image = (erf((x+1)/w2) - erf(x/w2))*(erf((y+1)/w2) - erf(y/w2))

    # Step 3: Normalize Gaussian to a maximum possible peak of A
    max_peak = (2*erf(0.5/w2))**2
    image *= A/max_peak

    # Step 4: Add an offset to simulate background light noise level
    return (image + offset).astype('uint16')


def Poisson_image(n, w=1, A=4095, c=(0, 0), offset=0):
    """
    Gaussian image with Poisson noise.

    Inputs
    ------
        n : tuple of ints
            Image size.

        w : float
            Gaussian width. Default 1.

        A : float
            Gaussian amplitude. Default is 4095 = 2**12 - 1.

        c : tuple of floats
            Gaussian center. Default is (0,0).

        offset : float
            The offset simulates background light noise level. Default is 0.

    Returns
    -------
      image : ndarray (dtype=uint16)
        A 2D image of shape (n[0],n[1]).
    """

    image = Gaussian_image(n, w=w, A=A, c=c, offset=offset)
    return poisson(image).astype('uint16')


def generate_synthetic_image_data_set(outdir, N_images, n, **kwargs):
    """
    Generates a data set of image files

    Inputs
    ------
        outdir : str
            Output directory name. If a directory with this name exists then it
            will be deleted (along with all files in it) and a new directory
            will be created.

        N_images : int
            Number of images in data set.

        n : tuple of ints
            Number of horizontal and vertical points in the images.

        kwargs : dict
            Keyword arguments.

    Keyword arguments
    -----------------
        A : float
            Maximum pixel value in the images. Default is 10700.

        w : float
            Gaussian width for the images. Default is 10.

        offset : float
            Image offset (see Gaussian_image and Poisson_image). Default is 10.

        spread_factor : float
            Maximum displacement (0.95 means up to 95% of the image edges).
            Default is 0.95.

        prefix : str
            Prefix output files names with this string. Default is "image_".

        displacements_filename : str
            Displacements output file name. Default is "displacements.txt".

        imdir : str
            The image output directory. Default is
            "outdir/images_w"+str(w)+"_o"+str(offset).

    Returns
    -------
        imdir : str
            The directory to which the images were output.
    """

    # Step 1: Set keyword arguments
    valid_keys = ("A", "w", "offset", "spread_factor", "prefix",
                  "displacements_filename", "imdir")
    for key in kwargs:
        if key not in valid_keys:
            raise ValueError(
                f"Unknown keyword argument {key}. Valid arguments are {valid_keys}")
    A = kwargs.get("A", 10700)
    w = kwargs.get("w", 10)
    offset = kwargs.get("offset", 10)
    spread_factor = kwargs.get("spread_factor", 0.95)
    prefix = kwargs.get("prefix", "image_")
    imid = "_w"+str(w)+"_o"+str(offset)
    displacements_filename = \
        kwargs.get("displacements_filename", "displacements"+imid+".txt")
    imdir = \
        kwargs.get("imdir", pjoin(outdir, "images"+imid))

    # Step 2: Create the output directory
    print(f"(RETINAS) Creating output directory {outdir}")
    rmtree(outdir, ignore_errors=True)
    print(f"(RETINAS) Creating image output directory {imdir}")
    makedirs(imdir)

    # Step 3: Generate the images and displacements.txt files
    print("(RETINAS) Beginning image generation")
    start = time()
    with open(pjoin(outdir, displacements_filename), "w", encoding="ascii") as f:
        c = (0, 0)
        f.write(f"{c[0]:.15e} {c[1]:.15e}\n")
        Poisson_image((n[0], n[1]), A=A, w=w, c=c).tofile(
            pjoin(imdir, prefix + "01.bin"), format="%u")
        for i in range(1, N_images+1):
            c = (spread_factor*(random()-0.5) *
                 n[0], spread_factor*(random()-0.5)*n[1])
            f.write(f"{c[0]:.15e} {c[1]:.15e}\n")
            Poisson_image((n[0], n[1]), A=A, w=w, c=c).tofile(
                pjoin(imdir, prefix + f"{i+1:02d}.bin"), format="%u")
            if not i % (N_images/5):
                print(
                    f"(RETINAS) Finished generating image {i:05d} out of {N_images:05d}")

    # All done!
    print(
        f"(RETINAS) Finished generating {N_images} images of size {n} in {time()-start:.1f} seconds")
    return imdir


def rebin(im, f, dtype=uint16):
    """
    Rebins an image by a factor f. If the original image has shape (m, n), then
    the rebinned image will have size (m/f, n/f). We do not check whether or not
    f divides m and n. The new "superpixels" in the rebinned image contain the
    sum of regions of size (f, f).

    Inputs
    ------
      im : ndarray
        Input image if size (m, n).

      f : int
        Rebinning factor.

    Returns
    -------
      im_out : ndarray
        Output image if size (m/f, n/f).
    """

    shape = (im.shape[0]//f, f, im.shape[1]//f, f)
    return im.reshape(shape).sum(axis=(-1, 1), dtype=dtype)


def freq_shift(new_frame_freq, horizontal_translation, vertical_translation, N_horizontal, N_vertical):
    """
    .------------.
    | freq_shift |
    .------------.

    This function shifts a given frame onto our eigenframe in the Fourier domain.

    Function inputs:
        - new_frame_freq    : Current frame data in Fourier Domain
        - nz                : Number of pixels in z-direction
        - ny                : Number of pixels in y-direction
    Function output:
        - the current (present) frame shifted onto our currentz eigenframe
    """
    # Initialize shift arrays.
    shift_horizontal = []
    shift_vertical = []

    # No negative in exp() since we're shifting *back* to the first frame.
    #             The if condition is used to match the weird array format in the Matlab code.
    #             Note also that Matlab starts index arrays at 1 rather than 0.
    for i_vertical in range(N_vertical):
        if i_vertical < N_vertical/2:
            shift_vertical.append(
                exp(2.*pi*complex(0, 1)*vertical_translation*(i_vertical/N_vertical)))
        else:
            shift_vertical.append(exp(
                2.*pi*complex(0, 1)*vertical_translation*((i_vertical-N_vertical)/N_vertical)))
    shift_vertical = array(shift_vertical)
    conjugate_index = int(N_vertical/2)
    shift_vertical[conjugate_index] = shift_vertical[conjugate_index].real

    for i_horizontal in range(N_horizontal):
        if i_horizontal < N_horizontal/2:
            shift_horizontal.append(
                exp(2.*pi*complex(0, 1)*horizontal_translation*(i_horizontal/N_horizontal)))
        else:
            shift_horizontal.append(exp(
                2.*pi*complex(0, 1)*horizontal_translation*((i_horizontal-N_horizontal)/N_horizontal)))
    shift_horizontal = array(shift_horizontal)
    conjugate_index = int(N_horizontal/2)
    shift_horizontal[conjugate_index] = shift_horizontal[conjugate_index].real

    # Again, on small examples the Matlab code is actually doing an outer product.
    shift_all = outer(shift_vertical, shift_horizontal)

    # Create the phase-shifted frame.
    shifted_frame = multiply(new_frame_freq, shift_all)
    return shifted_frame


def gpu_works(libpath: str):
    """ Given the library path, run a simple function to test if the GPU works. """

    lib = CDLL(libpath)
    lib.gpu_works.argtypes = []
    lib.gpu_works.restype = c_bool
    return lib.gpu_works()
