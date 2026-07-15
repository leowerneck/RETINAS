# C Implementation

## Purpose

Canonical owner for C-backend state, memory, precision, FFTW/CBLAS boundaries,
selected implementation units, and backend-local lifecycle evidence. Shared
frame flow remains in the [registration pipeline](../algorithms/registration-pipeline.md);
declaration or source presence alone does not establish an installed or working
public API.

## Ground Truth

| Evidence layer | Exact paths and symbols |
| --- | --- |
| State and declarations | `retinas/src/cross_correlation_c/retinas.h`: precision macros, `state_struct`, callable prototypes |
| Definitions and lifecycle | `retinas/src/cross_correlation_c/*.c`, especially `state_initialize`, `set_first_reference_image`, `initialize_image_sum`, `add_new_image_to_sum`, `update_reference_image_from_image_sum`, `state_finalize` |
| Build selection | `retinas/src/cross_correlation_c/meson.build`; `retinas/src/meson.build`; `meson.build.in` C dependency, precision, library, and install branches |
| Binding and caller | `retinas/retinas.py`: `initialize_library_functions`, constructor, online/offline composite methods, `finalize`; `retinas/utils.py`: `setup_library_function` |
| Example evidence | `retinas/synthetic_data_test.py`; offline pure-Python example in `README.md` does not execute C |
| History used to narrow current evidence | Git history introducing C offline units and later bridge lifecycle changes; current files remain authoritative |

Every source named by the C manifest has this observed role:

| Selected unit | Observed operation or helper |
| --- | --- |
| `state_initialize.c`; `state_finalize.c` | Allocate/free state and FFTW plans |
| `typecast_input_image_and_compute_brightness.c`; `typecast_input_image_rebin_4x4_and_compute_brightness.c` | Normal input conversion/brightness; separate 4x4 rebin variant |
| `set_first_reference_image.c`; `initialize_image_sum.c` | Install first reference; separately copy reference into accumulator |
| `cross_correlate_ref_and_new_images.c` | Forward transform, frequency product, inverse transform |
| `displacements_full_pixel_estimate.c`; `displacements_sub_pixel_estimate.c` | Full-pixel and localized sub-pixel estimates |
| `upsample_around_displacements.c` | Kernel construction and CBLAS complex matrix products |
| `compute_reverse_shift_matrix.c` | Frequency-domain reverse-shift factors |
| `update_reference_image.c`; `add_new_image_to_sum.c` | Aligned online update; aligned offline accumulation |
| `compute_displacements_and_update_reference_image.c`; `compute_displacements_and_add_new_image_to_sum.c` | Normal-mode composite entry points |
| `update_reference_image_from_image_sum.c` | Replace reference from accumulator and reset sum/counter |
| `get_reference_image_time.c`; `get_reference_image_freq.c` | Copy reference into caller-owned time/frequency output |

## Current Contract

### State, memory, precision, and dependencies

`state_initialize.c` copies dimensions, update coefficients, upsample factor,
shot-noise flag, and shift into a heap-allocated `state_struct`. It obtains
three reusable complex scratch arrays and four image arrays (`new_image_time`,
`new_image_freq`, `ref_image_freq`, and `image_sum_freq`) through
`FFTW_ALLOC_COMPLEX`, then creates forward and inverse 2-D plans. It does not
check allocation or plan results. `state_finalize.c` frees those seven
allocated arrays with `FFTW_FREE`, destroys both plans, and frees the struct.
The declared `reciprocal_new_image_time` pointer is not allocated, used, or
freed by current C units.

The three `aux_array*` allocations are reused as logical workspaces for image
products, correlation, kernels, matrix products, and shift factors. C has no
CUDA-style descriptive pointer aliases: operation-local meaning comes from each
caller. Call order therefore owns scratch validity.

`retinas.h` maps `REAL`, `COMPLEX`, FFTW routines, and CBLAS operations through
`PRECISION`. `meson.build.in` supplies `-DPRECISION=0` with `fftw3f` or
`-DPRECISION=1` with `fftw3`; its C branch checks FFTW allocation/plan/execute
symbols and complex GEMM/index-of-maximum BLAS symbols before selecting library
target `retinas`. Static selection is not compiled/linked evidence.

FFTW documents `fftw_alloc_complex` as an `fftw_malloc` convenience allocator,
and `fftw_malloc` as malloc-like rather than zero-initializing. This supports
the allocation reading below; repository writes still determine RETINAS state.
See [FFTW memory allocation](https://fftw.org/fftw3_doc/Memory-Allocation.html).

### Declaration, definition, selection, and binding

| Surface | Observed evidence |
| --- | --- |
| Normal bridge family | `state_initialize/finalize`, normal preprocessing, first reference, correlation, full/sub-pixel estimates, upsampling, online update, accumulation, average replacement, and both normal composites are declared in `retinas.h`, defined in selected units, and bound by `retinas/retinas.py` where applicable |
| Unbound declared helpers | Rebin preprocessing, reverse-shift construction, and time/frequency getters are declared, defined, and build-selected; bridge bindings are not found |
| Accumulator initializer | `initialize_image_sum` is defined and build-selected, but declaration and call site are not found |
| Shot-noise requests | Bridge conditionally requests `typecast_input_image_and_compute_brightness_shot_noise`, full/sub-pixel `_shot_noise` estimates, and online/offline `_shot_noise` composites; matching C declarations, definitions, and manifest units are not found. C state still declares a shot-noise flag and reciprocal pointer. Support or intended absence is unresolved |

The C manifest assigns `headers = files('retinas.h')`; `meson.build.in` installs
that variable when either native backend is enabled. No disposable-prefix
install was observed. Which declarations maintainers intend as supported public
surface, and intended C-only/CUDA-only/combined header behavior, remain
unresolved. Generated-origin warnings also leave header regeneration provenance
unresolved; follow [generated boundaries](../generated-boundaries.md).

### Offline accumulator chain

Static evidence gives this exact chain:

1. `state_initialize.c` allocates `image_sum_freq` but writes neither its
   elements nor `image_counter`.
2. `set_first_reference_image.c` writes only `ref_image_freq`.
3. Bridge `compute_displacements_and_add_new_image_to_sum` calls first-reference
   setup on the first offline call, clears its Python `first_image` flag, and
   returns before calling the native composite.
4. On the next offline call, `compute_displacements_and_add_new_image_to_sum.c`
   reaches `add_new_image_to_sum.c`, which uses `+=` on `image_sum_freq` and
   increments `image_counter`.
5. `initialize_image_sum.c` would copy the reference and set counter to one,
   but it is only build-selected: no header declaration, bridge binding, or
   repository call site is found.

This is an uninitialized-state access chain by static source and FFTW allocation
semantics. Runtime outcome, frequency, user impact, safety, and intended repair
remain unresolved without a controlled reproducer; it is not labeled a
confirmed runtime defect.

## Verification And Impact

Reconcile header prototypes, all 18 manifest entries, definitions, bridge symbol
requests, both precision branches, and closest callers. Current evidence is
`declared`, `defined`, `build-selected`, `bound`, or `selected-by-test/example`
as stated. No C build, load, execution, or assertion oracle was produced here;
`compiled/linked`, `executed`, and `assertion-covered` remain unresolved.

Smallest accumulator experiment: disposable C build in one named precision,
construct state, run first and second offline calls on a fixed image pair, and
inspect sum/counter under a memory-initialization tool with an explicit oracle.
Do not infer numerical parity from successful execution.

Changes to C headers, sources, manifest, root build selection, bridge bindings,
or synthetic caller require review here and in [registration pipeline](../algorithms/registration-pipeline.md),
[testing](../testing/index.md), [implementation hub](index.md), and
[change impact](../change-impact.md).
