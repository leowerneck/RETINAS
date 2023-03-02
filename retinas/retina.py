from numpy import exp, array, single, double
from ctypes import c_bool, c_uint16, c_int, c_float, c_double
from ctypes import c_void_p, POINTER, cast, CDLL

from utils import setup_library_function
from utils import center_array_max_return_displacements

class retina:
    """ Main code class """

    def finalize(self):
        """ Finalize the retina object """
        if self.initialized:
            self.lib_state_finalize(self.state)
            self.initialized = False

    def initialize_library_functions(self, libpath):
        """
        Initialize the library.

        Inputs
        ------
          libpath : str
            Path to the library.

          real : type
            Type of float to use (default is float).

          c_real : ctype
            Type of C float to use (default is c_float).

        Returns
        -------
          lib : ctypes.CDLL
            Library interface file, fully initialized.

        Raises
        ------
          ValueError : If precision has an unsupported value.
        """

        # Step 3.a: Load the library
        lib = CDLL(libpath)

        # Step 3.b: Set up all the functions in the library
        # Step 3.b.1: The state_initialize function
        # state_struct *state_initialize(
        #     const int N_horizontal,
        #     const int N_vertical,
        #     const REAL upsample_factor,
        #     const REAL A0,
        #     const REAL B1,
        #     const bool shot_noise_method,
        #     const REAL shift );
        setup_library_function(lib.state_initialize,
            [c_int, c_int, self.c_real, self.c_real, self.c_real, c_bool, self.c_real], c_void_p)
        self.lib_state_initialize = lib.state_initialize

        # Step 3.b.2: The state_finalize function
        # void state_finalize( state_struct *restrict state );
        setup_library_function(lib.state_finalize, [c_void_p], None)
        self.lib_state_finalize = lib.state_finalize

        # Step 3.b.3: The typecast_input_image_and_compute_brightness function
        # REAL typecast_input_image_and_compute_brightness(
        #     const uint16_t *restrict input_array,
        #     state_struct *restrict state );
        if self.shot_noise:
            setup_library_function(lib.typecast_input_image_and_compute_brightness_shot_noise,
                [self.c_uint16_p, c_void_p], self.c_real)
            self.lib_typecast_input_image_and_compute_brightness = \
                lib.typecast_input_image_and_compute_brightness_shot_noise
        else:
            setup_library_function(lib.typecast_input_image_and_compute_brightness,
                [self.c_uint16_p, c_void_p], self.c_real)
            self.lib_typecast_input_image_and_compute_brightness = \
                lib.typecast_input_image_and_compute_brightness

        # Step 3.b.4: The set_zeroth_eigenframe function
        # void set_zeroth_eigenframe( state_struct *restrict state );
        setup_library_function(lib.set_zeroth_eigenframe, [c_void_p], None)
        self.lib_set_zeroth_eigenframe = lib.set_zeroth_eigenframe

        # Step 3.b.5: The cross_correlate_ref_and_new_images function
        # void cross_correlate_ref_and_new_images(state_struct *restrict state);
        setup_library_function(lib.cross_correlate_ref_and_new_images, [c_void_p], None)
        self.lib_cross_correlate_ref_and_new_images = lib.cross_correlate_ref_and_new_images

        # Step 3.b.6: The displacements_full_pixel_estimate function
        # void displacements_full_pixel_estimate(
        #     state_struct *restrict state,
        #     REAL *restrict displacements );
        if self.shot_noise:
            setup_library_function(lib.displacements_full_pixel_estimate_shot_noise,
                [c_void_p, self.c_real_p], None)
            self.lib_displacements_full_pixel_estimate = \
                lib.displacements_full_pixel_estimate_shot_noise
        else:
            setup_library_function(lib.displacements_full_pixel_estimate,
                [c_void_p, self.c_real_p], None)
            self.lib_displacements_full_pixel_estimate = \
                lib.displacements_full_pixel_estimate

        # Step 3.b.7: The displacements_sub_pixel_estimate function
        # void displacements_sub_pixel_estimate(
        #     state_struct *restrict state,
        #     REAL *restrict displacements );
        if self.shot_noise:
            setup_library_function(lib.displacements_sub_pixel_estimate_shot_noise,
                [c_void_p, self.c_real_p], None)
            self.lib_displacements_sub_pixel_estimate = \
                lib.displacements_sub_pixel_estimate_shot_noise
        else:
            setup_library_function(lib.displacements_sub_pixel_estimate,
                [c_void_p, self.c_real_p], None)
            self.lib_displacements_sub_pixel_estimate = \
                lib.displacements_sub_pixel_estimate


        # Step 3.b.8: The upsample_and_compute_subpixel_displacements function
        # void upsample_around_displacements(
        #     state_struct *restrict state,
        #     REAL *restrict displacements );
        setup_library_function(lib.upsample_around_displacements,
            [c_void_p, self.c_real_p], None)
        self.lib_upsample_around_displacements = lib.upsample_around_displacements

        # Step 3.b.9: The build_next_eigenframe function
        # void build_next_eigenframe(
        #     const REAL *restrict displacements,
        #     state_struct *restrict state );
        setup_library_function(lib.build_next_eigenframe,
            [self.c_real_p, c_void_p], None)
        self.lib_build_next_eigenframe = lib.build_next_eigenframe

        # Step 3.b.10: The compute_displacements_and_build_next_eigenframe function
        # void compute_displacements_and_build_next_eigenframe(
        #     state_struct *restrict state,
        #     REAL *restrict displacements );
        if self.shot_noise:
            setup_library_function(lib.compute_displacements_and_build_next_eigenframe_shot_noise,
                       [c_void_p, self.c_real_p], None)
            self.lib_compute_displacements_and_build_next_eigenframe = \
                lib.compute_displacements_and_build_next_eigenframe_shot_noise
        else:
            setup_library_function(lib.compute_displacements_and_build_next_eigenframe,
                       [c_void_p, self.c_real_p], None)
            self.lib_compute_displacements_and_build_next_eigenframe = \
                lib.compute_displacements_and_build_next_eigenframe

    def __init__(self, libpath, N_horizontal, N_vertical, upsample_factor,
                 time_constant, precision="single", shot_noise=False, shift=0):
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
            constant (x) and recursion coeffs (A0, B1).

          precision : str
            Code precision (default="single").

          time_constant : float
            Time constant for IIR filter, used to calculate decay
            constant (x) and recursion coeffs (A0, B1).

          shot_noise : boolean
            Whether or not to use the shot noise method (default=False).

          shift : float
            Amount to shift new images before taking their reciprocal
            when the shot noise method is enabled (default=0).
        """

        self.initialized = False

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

        # Step 3.b: Initialize additional parameters
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
        self.shot_noise      = shot_noise
        self.shift           = shift if shot_noise else -1

        # Step 3.c: Initialize library functions
        self.initialize_library_functions(libpath)

        # Step 3.d: Initialize the state object
        self.state = self.lib_state_initialize(
                       self.N_horizontal,
                       self.N_vertical,
                       self.upsample_factor,
                       self.A0,
                       self.B1,
                       self.shot_noise,
                       self.shift)

    def __del__(self):
        """ Class destructor """
        self.finalize()

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

            brightness = self.lib_typecast_input_image_and_compute_brightness(
                cast(new_image.ctypes.data, self.c_uint16_p), self.state)

            self.lib_set_zeroth_eigenframe(self.state)
            self.first_image = False

            return array([0,0]) #array([h_0, v_0])

        brightness = self.lib_typecast_input_image_and_compute_brightness(
            cast(new_image.ctypes.data, self.c_uint16_p), self.state)

        displacements = array([0, 0], dtype=self.real)
        self.lib_cross_correlate_and_compute_displacements(
            self.state, cast(displacements.ctypes.data, self.c_real_p))

        return displacements

    def compute_displacements_wrt_ref_image_and_build_next_eigenframe(self, new_image):
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

            brightness = self.lib_typecast_input_image_and_compute_brightness(
                cast(new_image.ctypes.data, self.c_uint16_p), self.state)

            self.lib_set_zeroth_eigenframe(self.state)
            self.first_image = False

            return array([h_0, v_0])

        brightness = self.lib_typecast_input_image_and_compute_brightness(
            cast(new_image.ctypes.data, self.c_uint16_p), self.state)

        displacements = array([0, 0], dtype=self.real)
        self.lib_compute_displacements_and_build_next_eigenframe(
            self.state, cast(displacements.ctypes.data, self.c_real_p))

        return displacements
