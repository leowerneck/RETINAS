from os import makedirs
from os.path import join as pjoin
from shutil import rmtree
from time import time
from ctypes import c_int, POINTER
from numpy import unravel_index, argmax, rint, roll
from numpy import mgrid, sqrt, exp, ones, zeros, uint16
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

def Gaussian_image(n, w=1, A=4095, c=(0,0), offset=0):
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
    y, x = mgrid[0:n[1],0:n[0]]
    x = x - n[0]/2 - c[0]
    y = y - n[1]/2 - c[1]

    # Step 2: Set the image
    w2    = w/sqrt(2)
    image = ( erf((x+1)/w2) - erf(x/w2) )*( erf((y+1)/w2) - erf(y/w2) )

    # Step 3: Normalize Gaussian to a maximum possible peak of A
    max_peak = (2*erf(0.5/w2))**2
    image   *= A/max_peak

    # Step 4: Add an offset to simulate background light noise level
    image += offset

    return image.astype('uint16')

def Poisson_image(n, w=1, s=0.05, A=4095, c=(0,0), offset=0, background="dark", radius=0, dotradius=0):
    """
    Gaussian image with Poisson noise.

    Inputs
    ------
        n : tuple of ints
            Image size.

        w : float
            Gaussian width. Default 1.

        s : float
            Poisson noise strength. Default is 0.05.

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

def generate_synthetic_image_data_set(outdir, n, Nh, Nv,
                                      A=10700, w=10, offset=10, imdir=None,
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

    Returns
    -------
        imdir : str
            The directory to which the images were output.
    """

    # Step 1: Create the output directory
    print(f"(RETINA) Creating output directory {outdir}")
    rmtree(outdir, ignore_errors=True)
    if imdir is None:
        imdir = pjoin(outdir, "images_w"+str(w)+"_o"+str(offset))
    print(f"(RETINA) Creating image output directory {imdir}")
    makedirs(imdir)

    # Step 2: Generate the images and displacements.txt files
    print("(RETINA) Beginning image generation")
    start = time()
    with open(pjoin(outdir, "displacements.txt"), "w") as f:
        dh = 0
        dv = 0
        im = Poisson_image((Nh,Nv), A=A, w=w, c=(dh,dv))
        print(im.shape)
        im.tofile(pjoin(imdir, prefix + "01.bin"), format="%u")
        f.write(f"{dh:.15e} {dv:.15e}\n")
        for i in range(1,n+1):
            dh = spread_factor*(random()-0.5)*Nh
            dv = spread_factor*(random()-0.5)*Nv
            im = Poisson_image((Nh,Nv), A=A, w=w, c=(dh,dv))
            im.tofile(pjoin(imdir, prefix + f"{i+1:02d}.bin"), format="%u")
            f.write(f"{dh:.15e} {dv:.15e}\n")
            if not i%(n/5):
                print(f"(RETINA) Finished generating image {i:05d} out of {n:05d}")

    # All done!
    end = time()
    print(f"(RETINA) Finished generating {n} images of size {Nh} x {Nv} in {end-start:.1f} seconds")
    return imdir
