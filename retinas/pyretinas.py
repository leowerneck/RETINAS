"""
This is the Python version of RETINAS. This version was written by Leo Werneck
and contains contributions by Ector Ayala, Austin Branderberger, Zach Etienne,
Tyler Knowles, Megan Nolan, and Brian D'Urso.
"""

from typing import Union
from numpy import ndarray, double, uint16, float64
from numpy import abs, arange, unravel_index, argmax, argmin, exp, zeros, fix, pi
from numpy import fix, ceil, around, tensordot, stack, square, reciprocal, array
from numpy.fft import fftfreq, fft2, ifft2
from utils import freq_shift, center_array_max_return_displacements

class Pyretinas:
    """ Pyretinas class """

    def preprocess_new_image_cc(self, new_image) -> None:
        """
        Pre-process the new image for the cross-correlation algorithm.

        Inputs
        ------
          new_image : ndarray
            Array containing the new image.

        Returns
        -------
          Nothing. (self.new_image_freq is set)
        """
        self.new_image_freq = fft2(new_image)

    def preprocess_new_image_shot_noise(self, new_image) -> None:
        """
        Pre-process the new image for the cross-correlation algorithm.

        Inputs
        ------
          new_image : ndarray
            Array containing the new image.

        Returns
        -------
          Nothing. (self.new_image_freq is set)
        """
        self.squared_new_image_freq    = fft2(square(new_image+self.shift, dtype=float))
        self.reciprocal_new_image_freq = fft2(reciprocal(new_image+self.shift, dtype=float))

    def set_zeroth_eigenframe_cc(self) -> None:
        """
        Set the zeroth eigenframe in the cross-correlation algorithm.

        Inputs
        ------
          None.

        Returns
        -------
          Nothing. (self.ref_image_freq is set)
        """
        self.ref_image_freq = self.new_image_freq

    def set_zeroth_eigenframe_shot_noise(self) -> None:
        """
        Set the zeroth eigenframe in the shot noise algorithm.

        Inputs
        ------
          None.

        Returns
        -------
          Nothing. (self.ref_image_freq is set)
        """
        self.reciprocal_ref_image_freq = self.reciprocal_new_image_freq

    def cross_correlate_ref_and_new_images_cc(self) -> None:
        """
        Compute the cross-correlation between the reference and new images.

        Inputs
        ------
          Nothing.

        Returns
        -------
          Nothing. Result in stored in self.cross_correlation.
        """

        self.image_product     = self.new_image_freq * self.ref_image_freq.conj()
        self.cross_correlation = ifft2(self.image_product)

    def cross_correlate_ref_and_new_images_shot_noise(self) -> None:
        """
        Compute the cross-correlation between the reference and new images.

        Inputs
        ------
          Nothing.

        Returns
        -------
          Nothing. Result in stored in self.cross_correlation.
        """

        self.image_product     = self.squared_new_image_freq * self.reciprocal_ref_image_freq.conj()
        self.cross_correlation = ifft2(self.image_product)

    def displacements_full_pixel_estimate_cc(
            self, displacements : Union[list, ndarray]) -> None:
        """
        Compute a full-pixel estimate of the displacements by finding the maxima
        of the cross-correlation.

        Inputs
        ------
          displacements : list, tuple, or ndarray
            Array of length 2 storing the horizontal and vertical displacements.

        Returns
        -------
          Nothing.
        """

        maxima = unravel_index(argmax(abs(self.cross_correlation)),
                               self.cross_correlation.shape)

        displacements[0] = maxima[1]
        displacements[1] = maxima[0]
        if displacements[0] > self.N_horizontal/2:
            displacements[0] -= self.N_horizontal
        if displacements[1] > self.N_vertical/2:
            displacements[1] -= self.N_vertical

    def displacements_full_pixel_estimate_shot_noise(
            self, displacements : Union[list, ndarray]) -> None:
        """
        Compute a full-pixel estimate of the displacements by finding the minima
        of the cross-correlation.

        Inputs
        ------
          displacements : list, tuple, or ndarray
            Array of length 2 storing the horizontal and vertical displacements.

        Returns
        -------
          Nothing.
        """

        minima = unravel_index(argmin(abs(self.cross_correlation)),
                               self.cross_correlation.shape)

        displacements[0] = minima[1]
        displacements[1] = minima[0]
        if displacements[0] > self.N_horizontal/2:
            displacements[0] -= self.N_horizontal
        if displacements[1] > self.N_vertical/2:
            displacements[1] -= self.N_vertical

    def displacements_sub_pixel_estimate_cc(
            self, displacements : Union[list, ndarray]) -> None:
        """
        Compute a sub-pixel estimate of the displacements by finding the maxima
        of the upsampled cross-correlation.

        Inputs
        ------
          displacements : list, tuple, or ndarray
            Array of length 2 storing the horizontal and vertical displacements.

        Returns
        -------
          Nothing.
        """

        maxima = unravel_index(argmax(abs(self.upsampled_image)),
                               self.upsampled_image.shape)
        maxima = stack(maxima).astype(float64) - self.dftshift
        displacements[0] += maxima[1] / self.upsample_factor
        displacements[1] += maxima[0] / self.upsample_factor

    def displacements_sub_pixel_estimate_shot_noise(
            self, displacements : Union[list, ndarray]) -> None:
        """
        Compute a sub-pixel estimate of the displacements by finding the minima
        of the upsampled cross-correlation.

        Inputs
        ------
          displacements : list, tuple, or ndarray
            Array of length 2 storing the horizontal and vertical displacements.

        Returns
        -------
          Nothing.
        """

        minima = unravel_index(argmin(abs(self.upsampled_image)),
                               self.upsampled_image.shape)
        minima = stack(minima).astype(float64) - self.dftshift
        displacements[0] += minima[1] / self.upsample_factor
        displacements[1] += minima[0] / self.upsample_factor

    def upsample_around_displacements(
            self, displacements : Union[list, ndarray]) -> None:
        """
        Upsample image product around the displacement.

        Inputs
        ------
          displacements : list or ndarray
            Full-pixel estimate of the horizontal and vertical displacements.

        Returns
        -------
          Nothing. The output if stored in self.upsampled_image.
        """

        # This is the scikit-image algorithm
        shape     = self.image_product.shape
        midpoints = array([fix(axis_size / 2) for axis_size in shape])
        shifts    = array(list(reversed(displacements)), dtype=float64)
        shifts = around(shifts * self.upsample_factor) / self.upsample_factor
        upsample_factor = array(self.upsample_factor, dtype=float64)
        sample_region_offset  = self.dftshift - shifts * self.upsample_factor
        self.upsampled_image = self.image_product.conj()
        upsampled_region_size = [self.upsampled_region, ] * self.upsampled_image.ndim
        im2pi                 = complex(0, 2 * pi)
        dim_properties        = list(zip(self.image_product.shape,
                                         upsampled_region_size,
                                         sample_region_offset))

        for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
            kernel = (arange(ups_size) - ax_offset)[:, None] * fftfreq(n_items, self.upsample_factor)
            kernel = exp(-im2pi * kernel)
            self.upsampled_image = tensordot(kernel, self.upsampled_image, axes=(1, -1))

    def build_next_eigenframe_cc(
            self, displacements : ndarray) -> None:
        """
        Build the next reference frame (eigenframe) by shifting the new image
        in frequency space and adding it to the current reference frame.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.
        """

        # Step 1: Shift the new frame onto the Eigenframe
        new_image_shifted = freq_shift(self.new_image_freq,
                                       displacements[0], displacements[1],
                                       self.N_horizontal, self.N_vertical)

        # Step 2: Build the next Eigenframe
        self.ref_image_freq = self.A0*new_image_shifted + self.B1*self.ref_image_freq

    def build_next_eigenframe_shot_noise(
            self, displacements : ndarray) -> None:
        """
        Build the next reference frame (eigenframe) by shifting the new image
        in frequency space and adding it to the current reference frame.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.
        """

        # Step 1: Shift the new frame onto the Eigenframe
        new_image_shifted = freq_shift(self.reciprocal_new_image_freq,
                                       displacements[0], displacements[1],
                                       self.N_horizontal, self.N_vertical)

        # Step 2: Build the next Eigenframe
        self.reciprocal_ref_image_freq = self.A0*new_image_shifted + \
                                         self.B1*self.reciprocal_ref_image_freq


    def compute_displacements_wrt_ref_image_and_build_next_eigenframe(
            self, new_image : ndarray) -> None:
        """
        Compute the displacements with respect to the reference image.

        Inputs
        ------
          new_image : NumPy array
            NumPy array containing the new image.

        Returns
        -------
          displacements : NumPy array
            NumPy array containing the displacements.
        """

        if self.first_image:
            self.first_image = False
            new_image, h_0, v_0 = \
                center_array_max_return_displacements(new_image)
            self.preprocess_new_image(new_image)
            self.set_zeroth_eigenframe()
            return array([h_0, v_0])

        displacements = array([0, 0], dtype=float64)
        self.preprocess_new_image(new_image)
        self.cross_correlate_ref_and_new_images()
        self.displacements_full_pixel_estimate(displacements)
        self.upsample_around_displacements(displacements)
        self.displacements_sub_pixel_estimate(displacements)
        self.build_next_eigenframe(displacements)

        return displacements

    def __init__(self,
                 N_horizontal : int,
                 N_vertical : int,
                 upsample_factor : float,
                 time_constant : float,
                 shot_noise : bool = False,
                 shift : float = -1) -> None:
        """
        Initialize the Pyretinas object and return it.

        Inputs
        ------
          N_horizontal : int
            Number of horizontal pixels in the images.

          N_vertical : int
            Number of vertical pixels in the images.

          upsample_factor : float
            Upsampling factor. Displacements will be computed to a precision of
            1/upsample_factor.

          time_constant : float
            Provides a time scale associated with the problem. From this value
            we compute the filter weights that are used update the eigenframe.

          shot_noise : bool
            Whether or not to use the shot noise algorithm (default=False).

          shift : float
            Amount to shift new images before taking their reciprocal
            when the shot noise method is enabled (default=0).

        """

        self.N_horizontal      = N_horizontal
        self.N_vertical        = N_vertical
        self.upsample_factor   = upsample_factor
        self.upsampled_region  = int(ceil(1.5*upsample_factor))
        self.dftshift          = fix(self.upsampled_region / 2.0)
        self.time_constant     = time_constant
        self.x                 = exp(-1/time_constant)
        self.A0                = 1-self.x
        self.B1                = self.x
        self.shot_noise        = shot_noise
        self.shift             = shift
        self.image_product     = zeros((N_vertical, N_horizontal), dtype=complex)
        self.cross_correlation = zeros((N_vertical, N_horizontal), dtype=complex)
        self.upsampled_image   = zeros((N_vertical, N_horizontal), dtype=complex)
        self.first_image       = True

        if not shot_noise:
            self.new_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)
            self.ref_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)

            self.preprocess_new_image               = self.preprocess_new_image_cc
            self.set_zeroth_eigenframe              = self.set_zeroth_eigenframe_cc
            self.cross_correlate_ref_and_new_images = self.cross_correlate_ref_and_new_images_cc
            self.displacements_full_pixel_estimate  = self.displacements_full_pixel_estimate_cc
            self.displacements_sub_pixel_estimate   = self.displacements_sub_pixel_estimate_cc
            self.build_next_eigenframe              = self.build_next_eigenframe_cc

        else:
            self.squared_new_image_freq    = zeros((N_vertical, N_horizontal), dtype=complex)
            self.reciprocal_new_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)
            self.reciprocal_ref_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)

            self.preprocess_new_image               = self.preprocess_new_image_shot_noise
            self.set_zeroth_eigenframe              = self.set_zeroth_eigenframe_shot_noise
            self.cross_correlate_ref_and_new_images = self.cross_correlate_ref_and_new_images_shot_noise
            self.displacements_full_pixel_estimate  = self.displacements_full_pixel_estimate_shot_noise
            self.displacements_sub_pixel_estimate   = self.displacements_sub_pixel_estimate_shot_noise
            self.build_next_eigenframe              = self.build_next_eigenframe_shot_noise

    def __del__(self) -> None:
        """ Pyretinas destructor (does nothing) """
        pass

    def finalize(self) -> None:
        """ Destructor wrapper """
        self.__del__()
