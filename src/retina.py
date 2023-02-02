from numpy import exp, array, single, double
from ctypes import c_float, c_double, c_uint16, POINTER, cast
from initialize_library import initialize_library
from utils import center_array_max_return_displacements

class retina:
    """ Main code class """

    def initialize(self):
        self.state = self.lib.state_initialize(
                       self.N_horizontal,
                       self.N_vertical,
                       self.upsample_factor,
                       self.A0,
                       self.B1)

    def finalize(self):
        self.lib.state_finalize(self.state)

    def __init__(self, libpath, N_horizontal, N_vertical,
                 upsample_factor, time_constant, precision="single"):
        """
          Class constructor

          Inputs
          ------
            libpath : str
              Path to the library

            N_horizontal : int
              Number of horizontal pixels in the images.

            N_vertical : int
              Number of vertical pixels in the images.

            upsample_factot : float
              Upsampling factor.

            time_constant : float
              Time constant for IIR filter, used to calculate decay
              constant (x) and recursion coeffs (A0, B1)

            precision : str
              Code precision (default "single")
        """

        # Step 3.a: Set C types
        if precision.lower() == "single":
            self.real     = single
            self.c_real   = c_float
            self.c_real_p = POINTER(c_float)
        elif precision.lower() == "double":
            self.real     = double
            self.c_real   = c_double
            self.c_real_p = POINTER(c_double)
        else:
            raise ValueError(f'Unsupported precision {precision}. Supported values are "single" and "double"')
        self.c_uint16_p   = POINTER(c_uint16)

        self.lib             = initialize_library(libpath, real=self.real, c_real=self.c_real)
        self.N_horizontal    = N_horizontal
        self.N_vertical      = N_vertical
        self.upsample_factor = upsample_factor
        self.time_constant   = time_constant
        self.x               = exp(-1/time_constant)
        self.A0              = 1-self.x
        self.B1              = self.x
        self.libpath         = libpath
        self.precision       = precision
        self.first_image     = True
        self.initialize()

    def compute_displacements_wrt_ref_image(self, new_image):
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
            new_image, h_0, v_0 = \
                center_array_max_return_displacements(new_image, real=self.real)

            brightness = self.lib.typecast_input_image_and_compute_brightness(
                cast(new_image.ctypes.data, self.c_uint16_p), self.state)

            self.lib.set_zeroth_eigenframe(self.state)
            self.first_image = False

            return array([h_0, v_0])

        brightness = self.lib.typecast_input_image_and_compute_brightness(
            cast(new_image.ctypes.data, self.c_uint16_p), self.state)

        displacements = array([0, 0], dtype=self.real)
        self.lib.cross_correlate_and_compute_displacements(
            self.state, cast(displacements.ctypes.data, self.c_real_p))

        return displacements
