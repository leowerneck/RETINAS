from numpy import unravel_index, argmax, rint, roll, mgrid, sqrt, exp
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

def Gaussian_image(n, w=1, A=1, c=(0,0), offset=0):
    """
      Normal distribution (Gaussian) image.

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
          The offset simulates background light noise level.

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
    image = A * exp( -0.5*(x/w)**2 - 0.5*(y/w)**2 )*2**12 + offset
    return image.astype('uint16')

def Poisson_image(n, w=1, s=0.05, A=1, c=(0,0), offset=0):
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
          The offset simulates background light noise level.

      Returns
      -------
        image : ndarray
          A 2D image of shape (n[0],n[1]).
    """

    image = Gaussian_image(n, w=w, A=A, c=c, offset=offset)
    noise = poisson(image)
    image = image + s*noise*image.max()/noise.max()
    return image.astype('uint16')
