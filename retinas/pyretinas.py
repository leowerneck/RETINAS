"""
This is the Python version of RETINAS. This version was written by Leo Werneck
and contains contributions by Ector Ayala, Austin Branderberger, Zach Etienne,
Tyler Knowles, Megan Nolan, and Brian D'Urso.
"""

from typing import Union
from numpy import ndarray, float64
from numpy import abs as npabs
from numpy import arange, unravel_index, argmax, argmin, exp, zeros, pi
from numpy import fix, ceil, around, tensordot, stack, square, reciprocal, array
from numpy.fft import fftfreq, fft2, ifft2
from utils import freq_shift, center_array_max_return_displacements

class Pyretinas:
    """ Pyretinas class """

    def preprocess_new_image_and_compute_brightness_cc(self, new_image) -> None:
        """
        Pre-process the new image for the cross-correlation algorithm.

        Inputs
        ------
          new_image : ndarray
            Array containing the new image.

        Returns
        -------
          brightness : float64
            Sum of all pixels in the new image.
        """

        if self.first_image:
            new_image, self.h_0, self.v_0 = \
                center_array_max_return_displacements(new_image)

        self.new_image_freq = fft2(new_image)
        return new_image.sum(dtype=float64)

    def preprocess_new_image_and_compute_brightness_shot_noise(self, new_image) -> None:
        """
        Pre-process the new image for the cross-correlation algorithm.

        Inputs
        ------
          new_image : ndarray
            Array containing the new image.

        Returns
        -------
          brightness : float64
            Sum of all pixels in the new image.
        """
        self.squared_new_image_freq    = fft2(square(new_image+self.offset, dtype=float64))
        self.reciprocal_new_image_freq = fft2(reciprocal(new_image+self.offset, dtype=float64))
        return new_image.sum(dtype=float64)

    def set_first_reference_image_cc(self) -> None:
        """
        Set the first reference image in the cross-correlation algorithm.

        Inputs
        ------
          None.

        Returns
        -------
          Nothing. (self.ref_image_freq is set)
        """
        self.ref_image_freq = self.new_image_freq

    def set_first_reference_image_shot_noise(self) -> None:
        """
        Set the first reference image in the shot noise algorithm.

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
          None.

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
          None.

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

        maxima = unravel_index(argmax(npabs(self.cross_correlation)),
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

        minima = unravel_index(argmin(npabs(self.cross_correlation)),
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

        maxima = unravel_index(argmax(npabs(self.upsampled_image)),
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

        minima = unravel_index(argmin(npabs(self.upsampled_image)),
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
        shifts    = array(list(reversed(displacements)), dtype=float64)
        shifts = around(shifts * self.upsample_factor) / self.upsample_factor
        upsample_factor = array(self.upsample_factor, dtype=float64)
        sample_region_offset  = self.dftshift - shifts * upsample_factor
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

    def update_reference_image_cc(
            self, displacements : ndarray) -> None:
        """
        Update the reference frame by shifting the new image in frequency space
        and adding it to the current reference frame.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.

        """

        new_image_shifted = freq_shift(self.new_image_freq,
                                       displacements[0], displacements[1],
                                       self.N_horizontal, self.N_vertical)

        self.ref_image_freq = self.A0*new_image_shifted + self.B1*self.ref_image_freq

    def update_reference_image_shot_noise(
            self, displacements : ndarray) -> None:

        """
        Update the reference image by shifting the new image in frequency space
        and adding it to the current reference frame.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.
        """

        new_image_shifted = freq_shift(self.reciprocal_new_image_freq,
                                       displacements[0], displacements[1],
                                       self.N_horizontal, self.N_vertical)

        self.reciprocal_ref_image_freq = self.A0*new_image_shifted + \
                                         self.B1*self.reciprocal_ref_image_freq

    def add_new_image_to_sum_cc(
            self, displacements : ndarray) -> None:
        """
        Shift the new image and add it to the running sum.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.
        """

        self.image_counter  += 1
        self.image_sum_freq += freq_shift(self.new_image_freq,
                                          displacements[0], displacements[1],
                                          self.N_horizontal, self.N_vertical)

    def add_new_image_to_sum_shot_noise(
            self, displacements : ndarray) -> None:

        """
        Shift the new image and add it to the running sum.

        Inputs
        ------
          displacements : array-like object
            Contains the displacements.

        Returns
        -------
          Nothing.
        """

        self.image_counter  += 1
        self.image_sum_freq += freq_shift(self.reciprocal_new_image_freq,
                                          displacements[0], displacements[1],
                                          self.N_horizontal, self.N_vertical)

    def update_reference_image_from_image_sum_cc(self) -> None:
        """
        Update the reference image based on the image sum. Then reset the image
        sum to the new reference image and reset the image counter to one.

        Inputs
        ------
          None.

        Returns
        -------
          Nothing.
        """

        self.ref_image_freq = self.image_sum_freq/self.image_counter
        self.image_sum_freq = self.ref_image_freq
        self.image_counter  = 1

    def update_reference_image_from_image_sum_shot_noise(self) -> None:
        """
        Update the reference image based on the image sum. Then reset the image
        sum to the new reference image and reset the image counter to one.

        Inputs
        ------
          None.

        Returns
        -------
          Nothing.
        """

        self.reciprocal_ref_image_freq = self.image_sum_freq/self.image_counter
        self.image_sum_freq            = self.reciprocal_ref_image_freq
        self.image_counter             = 1

    def compute_displacements_and_add_new_image_to_sum(self) -> ndarray:
        """
        Compute the displacements and add new image to the sum.

        Inputs
        ------
          None.

        Returns
        -------
          displacements : NumPy array
            NumPy array containing the displacements.
        """

        if self.first_image:
            self.first_image = False
            self.set_first_reference_image()
            if self.shot_noise:
                self.image_sum_freq = self.reciprocal_ref_image_freq
            else:
                self.image_sum_freq = self.ref_image_freq
            return array([self.h_0, self.v_0])

        displacements = array([0, 0], dtype=float64)
        self.cross_correlate_ref_and_new_images()
        self.displacements_full_pixel_estimate(displacements)
        self.upsample_around_displacements(displacements)
        self.displacements_sub_pixel_estimate(displacements)
        self.add_new_image_to_sum(displacements)

        return displacements


    def compute_displacements_and_update_ref_image(self) -> ndarray:
        """
        Compute the displacements with respect to the reference image.

        Inputs
        ------
          None.

        Returns
        -------
          displacements : NumPy array
            NumPy array containing the displacements.
        """

        if self.first_image:
            self.first_image = False
            self.set_first_reference_image()
            return array([self.h_0, self.v_0])

        displacements = array([0, 0], dtype=float64)
        self.cross_correlate_ref_and_new_images()
        self.displacements_full_pixel_estimate(displacements)
        self.upsample_around_displacements(displacements)
        self.displacements_sub_pixel_estimate(displacements)
        self.update_reference_image(displacements)

        return displacements

    def __init__(self,
                 N_horizontal : int,
                 N_vertical : int,
                 upsample_factor : float64,
                 time_constant : float64,
                 shot_noise : bool = False,
                 offset : float64 = -1) -> None:
        """
        Initialize the Pyretinas object and return it.

        Inputs
        ------
          N_horizontal : int
            Number of horizontal pixels in the images.

          N_vertical : int
            Number of vertical pixels in the images.

          upsample_factor : float64
            Upsampling factor. Displacements will be computed to a precision of
            1/upsample_factor.

          time_constant : float64
            Provides a time scale associated with the problem. From this value
            we compute the filter weights that are used update the reference_image.

          shot_noise : bool
            Whether or not to use the shot noise algorithm (default=False).

          offset : float64
            Amount to offset new images before taking their reciprocal
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
        self.offset            = offset
        self.image_product     = zeros((N_vertical, N_horizontal), dtype=complex)
        self.cross_correlation = zeros((N_vertical, N_horizontal), dtype=complex)
        self.upsampled_image   = zeros((N_vertical, N_horizontal), dtype=complex)
        self.image_sum_freq    = zeros((N_vertical, N_horizontal), dtype=complex)
        self.first_image       = True
        self.h_0               = 0
        self.v_0               = 0
        self.image_counter     = 1

        if not shot_noise:
            self.new_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)
            self.ref_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)

            self.preprocess_new_image_and_compute_brightness = \
                self.preprocess_new_image_and_compute_brightness_cc
            self.set_first_reference_image             = self.set_first_reference_image_cc
            self.cross_correlate_ref_and_new_images    = self.cross_correlate_ref_and_new_images_cc
            self.displacements_full_pixel_estimate     = self.displacements_full_pixel_estimate_cc
            self.displacements_sub_pixel_estimate      = self.displacements_sub_pixel_estimate_cc
            self.update_reference_image                = self.update_reference_image_cc
            self.add_new_image_to_sum                  = self.add_new_image_to_sum_cc
            self.update_reference_image_from_image_sum = self.update_reference_image_from_image_sum_cc

        else:
            self.squared_new_image_freq    = zeros((N_vertical, N_horizontal), dtype=complex)
            self.reciprocal_new_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)
            self.reciprocal_ref_image_freq = zeros((N_vertical, N_horizontal), dtype=complex)

            self.preprocess_new_image_and_compute_brightness = \
                self.preprocess_new_image_and_compute_brightness_shot_noise
            self.set_first_reference_image             = self.set_first_reference_image_shot_noise
            self.cross_correlate_ref_and_new_images    = self.cross_correlate_ref_and_new_images_shot_noise
            self.displacements_full_pixel_estimate     = self.displacements_full_pixel_estimate_shot_noise
            self.displacements_sub_pixel_estimate      = self.displacements_sub_pixel_estimate_shot_noise
            self.update_reference_image                = self.update_reference_image_shot_noise
            self.add_new_image_to_sum                  = self.add_new_image_to_sum_shot_noise
            self.update_reference_image_from_image_sum = self.update_reference_image_from_image_sum_shot_noise

    def finalize(self) -> None:
        """ Pyretinas destructor (does nothing) """
