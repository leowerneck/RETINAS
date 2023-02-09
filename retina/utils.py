from os import makedirs
from os.path import join as pjoin
from shutil import rmtree
from time import time
from numpy import unravel_index, argmax, rint, roll
from numpy import mgrid, sqrt, exp, ones, zeros
from numpy import savetxt, uint16
from numpy.random import random, poisson
from ctypes import c_int, POINTER

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
    Ctype    = type(c_int)
    Cptrtype = type(POINTER(c_int))
    if not isinstance(params, list):
        raise TypeError(f'params must be of type list, but got {type(params)} instead.')
    for param in params:
        if not isinstance(param, Ctype) and not isinstance(param, Cptrtype):
            raise TypeError(f'params must be a list of C types, but found {type(param)}.')
    if not isinstance(returns, Ctype) and returns is not None:
        raise TypeError(f'returns must be a C type or None, but got {type(returns)} instead.')

    # Step 2.b: Setup the types of function parameters and return value
    func.argtypes = params
    func.restype  = returns

def center_array_max_return_displacements(image_array, real=float):
    """ Fast recentering of an image array around the maximum pixel """

    shape   = image_array.shape
    ind_max = unravel_index(argmax(image_array, axis=None), shape)
    move_0  = int(rint(shape[0]/2 - (ind_max[0]+0.5)+0.1))
    move_1  = int(rint(shape[1]/2 - (ind_max[1]+0.5)+0.1))
    h_0     = real(-move_1)
    v_0     = real(-move_0)
    return roll(image_array, (move_0,move_1), axis=(0,1)), h_0, v_0

def Gaussian_image(n, w=1, A=1, c=(0,0), offset=0, background="dark", radius=0, dotradius=0):
    """
    Normal distribution (Gaussian) image. Supports both dark and
    bright background.

    Inputs
    ------
        n : tuple of ints
            Image size.

        w : float
            Gaussian width (default=1).

        A : float
            Gaussian amplitude (default=1).

        c : tuple of floats
            Gaussian center (default=(0,0)).

        offset : float
            The offset simulates background light noise level (default=0).

        background : str
            Choice between dark and bright backgrounds (default=dark).

        radius : float
            Hard particle radius (default=0, which disables it).

        dotradius : float
            Radius of contrasting dot at image center (default=0).

    Returns
    -------
        image : ndarray (dtype=uint16)
            A 2D image of shape (n[0],n[1]).
    """

    if not isinstance(n, tuple) or len(n) != 2:
        raise TypeError("Image size must be a tuple of length 2.")
    if not isinstance(c, tuple) or len(c) != 2:
        raise TypeError("Gaussian center must be a tuple of length 2.")

    y, x = mgrid[0:n[1],0:n[0]]
    x = x - n[0]/2 - c[0]
    y = y - n[1]/2 - c[1]

    idx = ones((n[1],n[0]), dtype=bool)
    if radius > 0:
        idx = idx & (x**2 + y**2 <= radius**2) & (x**2 + y**2 >= dotradius**2)

    if background.lower() == "dark":
        image       = zeros((n[1],n[0]))
        image[idx] += A * exp( -0.5*(x[idx]/w)**2 - 0.5*(y[idx]/w)**2 )*2**12 + offset
    elif background.lower() == "bright":
        image       = A*2**12*ones((n[1],n[0]))
        image[idx] -= (A * exp( -0.5*(x[idx]/w)**2 - 0.5*(y[idx]/w)**2 )*2**12 + offset)
    else:
        ValueError(f"Unsupported background {background}. Valid options are \"dark\" and \"bright\".")

    return image.astype('uint16')

def Poisson_image(n, w=1, s=0.05, A=1, c=(0,0), offset=0, background="dark", radius=0, dotradius=0):
    """
    Poisson distribution image.

    Inputs
    ------
        n : tuple of ints
            Image size.

        w : float
            Gaussian width. Default 1.

        s : float
            Poisson noise strength. Default is 0.05.

        A : float
            Gaussian amplitude. Default is 1.

        c : tuple of floats
            Gaussian center. Default is (0,0).

        offset : float (dtype=uint16)
            The offset simulates background light noise level. Default is 0.

        background : str
            Choice between dark and bright backgrounds. Default is "dark".

        radius : float
            Hard particle radius. Default is 0, which disables it.

        dotradius : float
            Radius of contrasting dot at image center. Default is 0.

    Returns
    -------
      image : ndarray
        A 2D image of shape (n[0],n[1]).
    """

    image = Gaussian_image(n, w=w, A=A, c=c, offset=offset, background=background, radius=radius, dotradius=dotradius)
    noise = poisson(image)
    image = image + s*noise*image.max()/noise.max()
    return image.astype('uint16')

def generate_synthetic_image_data_set(outdir, n, Nh, Nv,
                                      spread_factor=0.95, prefix="image_",
                                      displacements_filename="displacements.txt"):
    """
    Generates a data set of image files

    Inputs
    ------
        outdir : str
            Output directory name. If a directory with this name exists then it
            will be deleted (along with all files in it) and a new directory
            will be created.

        n : int
            Number of images in data set.

        Nh : int
            Number of horizontal points in images.

        Nv : int
            Number of vertical points in images.

        spread_factor : float
            Maximum displacement (0.95 means up to 95% of the image edges).
            Default is 0.95.

        prefix : str
            Prefix output files names with this string. Default is "image_".

        displacements_filename : str
            Displacements output file name. Default is "displacements.txt".

    Returns
    -------
        Nothing.
    """

    # Step 1: Create the output directory
    print(f"(RETINA) Creating output directory {outdir}")
    rmtree(outdir, ignore_errors=True)
    imdir = pjoin(outdir, "images")
    makedirs(imdir)

    # Step 2: Generate the images and displacements.txt files
    print("(RETINA) Beginning image generation")
    start = time()
    with open(pjoin(outdir, "displacements.txt"), "w") as f:
        dh = 0
        dv = 0
        im = Poisson_image((Nh,Nv), c=(dh,dv))
        outfile = pjoin(imdir, prefix + "01.txt")
        savetxt(outfile, im, fmt="%u")
        f.write(f"{dh:.15e} {dv:.15e}\n")
        for i in range(1,n+1):
            dh = spread_factor*(random()-0.5)*Nh
            dv = spread_factor*(random()-0.5)*Nv
            im = Poisson_image((Nh,Nv), c=(dh,dv))
            outfile = pjoin(imdir, prefix + f"{i+1:02d}.txt")
            savetxt(outfile, im, fmt="%u")
            f.write(f"{dh:.15e} {dv:.15e}\n")

    # All done!
    end = time()
    print(f"(RETINA) Finished generating {n} images of size {Nh} x {Nv} in {end-start:.1f} seconds")
