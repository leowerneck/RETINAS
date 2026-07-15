# CUDA Implementation

## Purpose

Canonical owner for CUDA host/device boundaries, state allocation and aliases,
cuFFT/cuBLAS use, precision variants, selected units, and backend-local evidence
stages. Presence of kernels or host wrappers does not establish synchronization,
error handling, runtime success, or cross-backend parity.

## Ground Truth

| Evidence layer | Exact paths and symbols |
| --- | --- |
| State and precision | `retinas/src/cross_correlation_cuda/retinas.h`: `state_struct`, precision/library macros, alias fields |
| Declarations | `retinas/src/cross_correlation_cuda/function_prototypes.h`: host-callable functions and inline `CEXP` |
| Definitions | `retinas/src/cross_correlation_cuda/*.cu`: host wrappers, device kernels, lifecycle and variants |
| Build selection | CUDA `meson.build`; `retinas/src/meson.build`; `meson.build.in` CUDA dependency, precision, and `retinas_cuda` target |
| Bindings | `retinas/retinas.py`: common and mode-specific requests; `retinas/utils.py`: separate `gpu_works` binding |
| Example evidence | `retinas/synthetic_data_test.py`: normal-mode installed-path selection, no assertion oracle |

All 27 manifest units are accounted for:

| Selected units | Observed operation or helper |
| --- | --- |
| `state_initialize.cu`; `state_finalize.cu` | Host state/device allocation, aliases, cuFFT/cuBLAS handles, teardown |
| `typecast_input_image_and_compute_brightness.cu`; `typecast_input_image_and_compute_brightness_shot_noise.cu` | Host-to-device input, normal or reciprocal/squared preprocessing, cuBLAS brightness |
| `set_first_reference_image.cu` | Mode-aware first FFT plus accumulator copy/counter initialization |
| `cross_correlate_ref_and_new_images.cu`; `element_wise_multiplication_conj_2d.cu` | cuFFT correlation pipeline and device product kernel |
| `absolute_value_2d.cu`; `find_maxima_minima.cu` | Device magnitude helper; cuBLAS maximum/minimum index helpers |
| `displacements_full_pixel_estimate.cu`; `displacements_full_pixel_estimate_shot_noise.cu` | Normal maximum and shot-noise minimum full-pixel selection |
| `compute_kernels.cu`; `complex_conjugate_2d.cu`; `upsample_around_displacements.cu` | DFT kernels, device conjugation, and cuBLAS localized matrix products |
| `displacements_sub_pixel_estimate.cu`; `displacements_sub_pixel_estimate_shot_noise.cu` | Normal maximum and shot-noise minimum sub-pixel refinement |
| `compute_reverse_shift_matrix.cu`; `update_reference_image.cu` | Device shift factors and aligned IIR reference update |
| `add_new_image_to_sum.cu`; `update_reference_image_from_image_sum.cu` | Device accumulation and averaged-reference/reset operation |
| `compute_displacements_and_update_reference_image.cu`; `compute_displacements_and_update_reference_image_shot_noise.cu` | Normal and shot-noise online composites |
| `compute_displacements_and_add_new_image_to_sum.cu`; `compute_displacements_and_add_new_image_to_sum_shot_noise.cu` | Normal and shot-noise offline composites |
| `get_reference_image.cu`; `gpu_works.cu`; `print_arrays.cu` | Reference device-to-host retrieval, explicit GPU probe, diagnostic printers |

## Current Contract

### Host/device state and aliases

`state_initialize.cu` allocates `state_struct` on the host, initializes scalar
configuration and `image_counter = 0`, then uses `cudaMalloc` for integer/real/
complex scratch and image arrays. It creates one 2-D cuFFT plan and one cuBLAS
handle. Return codes from allocation and handle/plan creation are not checked in
this function.

Only ten device pointers own allocations. Descriptive fields alias them:

- `aux_array1`: `ref_image_time`, `image_product`, `upsampled_image`,
  `horizontal_shifts`;
- `aux_array2`: `cross_correlation`, `horizontal_kernel`, `vertical_kernel`,
  `vertical_shifts`;
- `aux_array3`: `partial_product`, `shift_matrix`.

`state_finalize.cu` frees the ten owners, not aliases, destroys the cuFFT plan
and cuBLAS handle, then frees host state. `cudaMalloc` memory is not cleared.
CUDA `set_first_reference_image.cu` enqueues a kernel that copies
reference-frequency elements to `image_sum_freq`, then sets the host counter to
one. No synchronization or error check proves that the device copy completed
before bridge first-offline return. See
[CUDA memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
and [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

### Host API, device work, and libraries

`function_prototypes.h` marks declared callable routines `__host__`; many
definitions use `extern "C" __host__`. Internal transformation loops are
mostly `static __global__` kernels launched by host wrappers. NVIDIA defines
`__global__` as device-executed kernel code and host launches as asynchronous;
therefore a launch in source is not completion/error proof. See
[CUDA execution-space specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-execution-space-specifiers).

cuFFT owns complex 2-D plan/execution in first-reference, correlation,
shot-noise representation changes, and retrieval. Its official
[`cufftPlan2d` contract](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan2d)
describes plan dimensions and result statuses. cuBLAS owns brightness sums,
maximum/minimum indices, and complex matrix products. Its
[context contract](https://docs.nvidia.com/cuda/cublas/index.html#cublas-context)
requires a created handle and later destruction. RETINAS source currently does
not inspect these APIs' returned statuses except `gpu_works`, which explicitly
synchronizes and reads the last CUDA error.

Precision is compile-time: `PRECISION=0` maps to float/cuFloatComplex/C2C and
single cuBLAS calls; `PRECISION=1` maps to double/cuDoubleComplex/Z2Z and double
calls; other values trigger a header error. `meson.build.in` passes the same
precision flag and selects CUDA 10+ with cuBLAS/cuFFT for library
`retinas_cuda`. This is build configuration, not observed compilation.

### Variants and evidence stages

Normal and shot-noise preprocessing, estimates, and composites have separate
selected definitions. Shot-noise preprocessing stores reciprocal input and
squared input; first-reference transforms the reciprocal, correlation uses the
squared representation, then composite update/accumulation transforms the
reciprocal into `new_image_freq`. Normal estimates choose maxima; shot-noise
estimates choose minima.

Most host functions are declared, defined, and build-selected. Current header
does not declare either offline composite, the shot-noise online composite, or
`gpu_works`; those are nevertheless defined in selected units. Bridge requests
all three composites by mode, while `utils.gpu_works` separately requests the
probe. Declaration and binding evidence therefore differ.

`retinas.py` requests normal and shot-noise symbol families according to its
flag. `synthetic_data_test.py` selects both installed native paths but sets
shot noise false. The CUDA headers are not assigned to the `headers` install
variable; current root install declaration instead receives the C header.
Actual installed CUDA header surface and intended public symbols are
unresolved without maintainer intent and a disposable-prefix install.

| Evidence stage | Current CUDA result |
| --- | --- |
| `declared` | Prototype header covers most host operations; omissions are listed above |
| `defined` / `build-selected` | All 27 manifest units have definitions/helpers accounted for; configuration selects them only when `with-cuda` is enabled |
| `bound` | Bridge requests common and mode-specific registration symbols; utility wrapper requests `gpu_works` |
| `compiled/linked` | Unresolved; no build was run |
| `selected-by-test/example` | Manual comparison selects normal CUDA bridge path with fixed installed-library path |
| `executed` / `assertion-covered` | Unresolved; no runtime was run and no explicit comparison assertion/tolerance is present |

## Verification And Impact

Static reconciliation covered both headers, every manifest unit, definitions,
bridge requests, aliases, both precision branches, and normal/shot-noise flow.
Evidence reaches `declared`, `defined`, `build-selected`, `bound`, and
`selected-by-test/example` only where stated. No CUDA build, device probe,
execution, or assertion run is recorded for a named environment here. Those
layers remain unresolved; an unavailable toolchain or device is an environment
gap, not product failure.

Changes to CUDA headers, any `.cu` unit, manifest, precision/dependency branch,
bridge binding, or GPU helper require review here and in
[registration pipeline](../algorithms/registration-pipeline.md),
[testing](../testing/index.md), [implementation hub](index.md),
[generated boundaries](../generated-boundaries.md), and
[change impact](../change-impact.md). Runtime claims require named GPU,
toolchain, precision, mode, fixture, error checks, and oracle.
