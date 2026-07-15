# Ctypes Bridge

## Purpose

Canonical owner for Python-to-native library loading, ABI declarations, array
boundary assumptions, symbol selection, wrapper state, and native lifetime.
Backend capability comparison belongs in the landed
[parity matrix](../implementations/parity-matrix.md); layered public/install
evidence belongs in the landed [public API map](../public-api-map.md), not this
bridge owner.

## Ground Truth

| Evidence layer | Exact paths and symbols |
| --- | --- |
| Bridge implementation | `retinas/retinas.py`: class `retinas`, `initialize_library_functions`, constructor, preprocess/composite/update/finalize methods |
| FFI helper | `retinas/utils.py`: `setup_library_function`, which assigns `argtypes` and `restype`; centering helpers |
| C ABI | `retinas/src/cross_correlation_c/retinas.h`, selected C definitions and C `meson.build` |
| CUDA ABI | CUDA `retinas.h`, `function_prototypes.h`, selected `.cu` definitions and CUDA `meson.build` |
| Caller | `retinas/synthetic_data_test.py`; offline API prose/example in `README.md` |
| Focused history | Git changes that added then later removed `self.initialized = True`, plus later offline-binding correction; current source is authoritative |

Python documents `ctypes` as low-level native access that bypasses Python memory
safety; `CDLL` loads a shared library, while function `argtypes` and `restype`
declare call conversion/prototypes. See the official
[`ctypes` reference](https://docs.python.org/3/library/ctypes.html).

## Current Contract

### Loading, backend selection, and ABI

Constructor accepts an explicit `libpath`; `initialize_library_functions`
passes it directly to `CDLL`. There is no backend enum, filename rule, library
discovery, or ABI/version probe. Backend is therefore selected by caller-supplied
library contents. Precision is also caller-supplied and must match native build:
`single` chooses NumPy `single`, `c_float`, and `POINTER(c_float)`; `double`
chooses corresponding double types; other values raise `ValueError` before
loading.

`initialize_library_functions` keeps each requested native function on `self`
but does not retain the local `CDLL` object as a named bridge attribute or
return it, despite its docstring's return section. The bridge catches no library
load, symbol lookup, or native-call exceptions. `setup_library_function` raises
its own `TypeError` for unsupported type-descriptor shapes; it does not validate
the native library's actual ABI.

`setup_library_function` validates that requested parameter/return descriptors
are ctypes scalar or pointer classes, then sets native function metadata. Bridge
bindings are:

| Requested operation | `argtypes` | `restype` and selection |
| --- | --- | --- |
| `state_initialize` | two `c_int`, four selected real scalars around one `c_bool` | `c_void_p` |
| `state_finalize` | `c_void_p` | `None` |
| preprocessing | `POINTER(c_uint16)`, `c_void_p` | selected real; normal or `_shot_noise` symbol |
| first reference; correlation | `c_void_p` | `None` |
| full/sub-pixel estimate | `c_void_p`, selected-real pointer | `None`; normal or `_shot_noise` symbol |
| upsample | `c_void_p`, selected-real pointer | `None` |
| update reference; add to sum | selected-real pointer, `c_void_p` | `None` |
| online/offline composites | `c_void_p`, selected-real pointer | `None`; normal or `_shot_noise` symbol |
| averaged-reference replacement | `c_void_p` | `None` |

The comment above averaged-reference binding mentions a displacement pointer,
but its configured signature matches current C/CUDA one-state-pointer
definitions. Bridge does not bind native reference getters, C rebin
preprocessing, CUDA diagnostic printers, or CUDA `gpu_works`; the probe has a
separate wrapper in `retinas/utils.py`.

All selected symbols are accessed before native state creation. Consequently a
library lacking any symbol requested for the chosen mode cannot finish bridge
initialization. Current C sources provide the normal requested family; the five
shot-noise names requested by bridge are
`typecast_input_image_and_compute_brightness_shot_noise`,
`displacements_full_pixel_estimate_shot_noise`,
`displacements_sub_pixel_estimate_shot_noise`,
`compute_displacements_and_update_reference_image_shot_noise`, and
`compute_displacements_and_add_new_image_to_sum_shot_noise`; none is found in C
declarations, definitions, or selected units. CUDA selected definitions provide
both mode families, although its prototype header omits
`compute_displacements_and_update_reference_image_shot_noise` and both
`compute_displacements_and_add_new_image_to_sum*` composites. These are
symbol-layer facts, not support/parity conclusions.

### NumPy memory boundary

`preprocess_new_image_and_compute_brightness` takes `new_image.ctypes.data`,
casts its address to `POINTER(c_uint16)`, and passes native state. It does not
check dtype, two-dimensional shape, dimensions against `N_horizontal` and
`N_vertical`, element count, byte order, C contiguity, stride, alignment,
writeability, or lifetime across the call. Native implementations read
`N_horizontal * N_vertical` consecutive `uint16_t` values; callers therefore
must provide compatible, live contiguous storage. Optional first-image
centering uses NumPy helpers and replaces the local array with their rolled
result before taking the pointer, but this is not general validation.

Composite methods allocate a two-element NumPy displacement array with selected
real dtype and pass its contiguous data pointer. First-call returns instead use
`array([h_0, v_0])` without `dtype=self.real`, so that first result's dtype is
not guaranteed to match later native-result dtype. Brightness is converted by
ctypes according to selected `restype`.

### Construction, first calls, and lifetime

Defaults are `precision="single"`, `shot_noise=True`, `offset=0`, and
`center_first_image='max'`. Constructor docstring says shot noise defaults
false, while signature sets true. With a time constant, it computes
`A0 = 1 - exp(-1/time_constant)` and `B1 = exp(-1/time_constant)`; with `None`,
both become `-1`. Offset becomes `-1` when shot noise is false. State creation
receives dimensions, upsample factor, coefficients, flag, and effective offset;
returned pointer is stored without a null check.

Both high-level composite methods require prior preprocessing to populate native
current-image state. On their first call they invoke only
`set_first_reference_image`, set Python `first_image` false, and return centering
offsets. Later online/offline calls invoke their respective native composites.
For offline mode this produces backend-specific state chains:

- C first-reference setup does not initialize image sum/counter; its separate
  selected initializer is neither bound nor called, so second offline composite
  reaches accumulation against state not initialized by visible bridge flow.
- CUDA first-reference setup enqueues a reference-to-sum copy and sets the host
  counter to one. Default-stream ordering is visible, but completion before the
  bridge returns is unresolved because this path does not synchronize or check
  launch status.

`self.initialized` starts false. Current constructor loads functions and creates
native state but never changes it to true. `finalize` calls native finalization
only when that flag is true; `__del__` only calls `finalize`. Git history shows
the true assignment was once added for finalization and later removed while
offline CUDA bindings changed, but history does not override current behavior
or establish present intent. Under visible unmodified lifecycle, native
finalization is therefore not reached. Runtime resource impact and desired
lifecycle remain unresolved.

## Verification And Impact

Static verification compared every `setup_library_function` request with both
headers, definitions, manifests, precision types, and first/next-call wrapper
branches. Evidence is `bound`; corresponding native stages differ by backend as
described. Manual comparison script is `selected-by-test/example` for normal
mode only. No library was built, loaded, executed, or assertion-tested here.

Smallest safe experiments: use disposable matching-precision native builds;
verify initialization failure for a deliberately missing symbol; pass validated
contiguous `uint16` fixtures; instrument native finalization; and separately
exercise first/second offline calls with explicit sum/counter and numerical
oracles. Do not combine those results into parity without named backend, mode,
precision, fixture, tolerance, and environment.

Changes to `retinas/retinas.py`, FFI helper, native prototypes/definitions,
library names, precision, or caller array construction require review here and
in [registration pipeline](../algorithms/registration-pipeline.md),
[implementation hub](../implementations/index.md), [testing](../testing/index.md),
and [change impact](../change-impact.md). Review current public/install evidence
in [public API map](../public-api-map.md) and cross-backend synthesis in
[parity matrix](../implementations/parity-matrix.md).
