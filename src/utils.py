from numpy import unravel_index, argmax, rint, roll, mgrid, sqrt, exp, ones, zeros
from numpy.random import poisson

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
        Gaussian width (default=1).
      s : float
        Poisson noise strength (defaut=0.05).
      A : float
        Gaussian amplitude (default=1).
      c : tuple of floats
        Gaussian center (default=(0,0)).
      offset : float (dtype=uint16)
        The offset simulates background light noise level (default=0).
      background : str
        Choice between dark and bright backgrounds (default=dark).
      radius : float
        Hard particle radius (default=0, which disables it).
      dotradius : float
        Radius of contrasting dot at image center (default=0).

    Returns
    -------
      image : ndarray
        A 2D image of shape (n[0],n[1]).
    """

    image = Gaussian_image(n, w=w, A=A, c=c, offset=offset, background=background, radius=radius, dotradius=dotradius)
    noise = poisson(image)
    image = image + s*noise*image.max()/noise.max()
    return image.astype('uint16')
