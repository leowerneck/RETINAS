# Pure-Python Implementation

## Purpose

Own `Pyretinas` construction, NumPy state, mode dispatch, callable surface, and
lifecycle behavior. It also owns the standalone pure-Python `rebin` callable,
which is not a registration stage and has no current repository caller. Shared
transform and estimator contracts stay in the [algorithm owners](../algorithms/index.md);
this page does not claim native parity.

## Ground Truth

| Evidence | Exact paths and symbols |
| --- | --- |
| Implementation | `retinas/pyretinas.py`: `Pyretinas`, all `*_cc` and `*_shot_noise` stage methods, both composite compute methods, `__init__`, `finalize` |
| Local helpers | `retinas/utils.py`: `freq_shift`, `center_array_max_return_displacements`, `center_array_min_return_displacements`, `rebin` |
| Demonstrated callers | `retinas/synthetic_data_test.py`; offline example in `README.md` |
| Build/import boundary | `meson.build.in`, `retinas/meson.build`: no Python-module install declaration found |
| History for disputed reset prose | Git history of `retinas/pyretinas.py` and the `README.md` offline example; current source remains runtime authority |

## Current Contract

### Construction And State

`Pyretinas(N_horizontal, N_vertical, upsample_factor, time_constant=None,
shot_noise=False, offset=-1, center_first_image='max')` stores dimensions and
mode without input validation. Its signature defaults `offset` to `-1`, while
the constructor docstring says the default is `0`; intended default remains
unresolved. It sets:

- `upsampled_region = ceil(1.5 * upsample_factor)` and `dftshift` to truncation
  toward zero of half that region;
- `first_image=True`, centering offsets `h_0=v_0=0`, `image_counter=1`, and a
  complex128 zero `image_sum_freq` of shape
  `(N_vertical, N_horizontal)`;
- complex128 work arrays for product, correlation, and upsampled data; and
- when `time_constant` is not `None`, `x=exp(-1/time_constant)`, `A0=1-x`, and
  `B1=x`. With `None`, those three attributes are not created, so a later online
  update is unresolved until called and then reaches missing `A0`/`B1`.

Normal mode allocates `new_image_freq` and `ref_image_freq`. Shot-noise mode
instead allocates squared-current, reciprocal-current, and reciprocal-reference
arrays. NumPy FFT calls subsequently replace current-transform arrays with
complex128 results.

### Dispatch And Callable Surface

Construction binds eight generic instance attributes to either normal or
shot-noise bound methods: preprocessing, first-reference setup, correlation,
full- and sub-pixel estimates, online update, sum addition, and averaged
replacement. `upsample_around_displacements` and both composite methods are
shared. Changing `shot_noise` after construction does not rebind them.

Python exposes every stage method as a callable class attribute; none is marked
private. `README.md` demonstrates preprocessing, offline composite, averaged
replacement, and `finalize`; `retinas/synthetic_data_test.py` demonstrates
preprocessing and online composite. A formal supported public/internal split is
not found and remains unresolved.

### Per-Frame Inputs, Outputs, And Mutation

Preprocessing accepts an object used as a two-dimensional NumPy array. It does
not enforce `uint16`, configured shape, dimensionality, or contiguity. First
preprocessing may roll a local array around its maximum/minimum and stores the
returned horizontal/vertical offsets. Normal mode stores `fft2(image)`;
shot-noise mode stores transforms of `(image + offset)^2` and
`1/(image + offset)`. Both return brightness as a NumPy `float64` sum of the
unshifted-value image; see
[preprocessing and correlation](../algorithms/preprocessing-and-correlation.md).

Callers must preprocess before a composite method; composites take no image and
consume current state. First composite call installs a reference and returns
`array([h_0, v_0])`, whose observed dtype is NumPy's inferred integer dtype.
Later calls return a length-two `float64` array ordered horizontal, vertical.
Correlation and refinement details belong to
[displacement estimation](../algorithms/displacement-estimation.md).

Online composite replaces reference with
`A0 * aligned_current + B1 * reference`. Offline composite keeps a sum and
counter; full transitions belong to
[reference update modes](../algorithms/reference-update-modes.md).

### Offline Aliasing, Reset, And Finalization

On first offline composite, reference is assigned directly from current
transform, then `image_sum_freq` is assigned directly from that reference.
There is no copy: sum and reference are the same ndarray. Later `+=` in
`add_new_image_to_sum_*` mutates that shared object, so subsequent correlations
observe the accumulated value as reference. `README.md` describes a fixed
reference; whether aliasing is intended is unresolved.

Averaged replacement assigns a newly divided reference, then sets
`image_sum_freq=None`, `image_counter=0`, while leaving `first_image=False`.
Next offline composite reaches in-place addition to `None` after estimating a
displacement and raises `TypeError`. No reinitialization branch exists. Desired
multi-cycle behavior is unresolved.

`finalize()` has an empty body. It does not clear arrays, invalidate state, or
prevent later calls. There is no `Pyretinas.__del__`.

### Dependency Boundary

`rebin(im, f, dtype=uint16)` is a standalone utility callable, not a
`Pyretinas` stage, and has no current repository caller. It reshapes the input
into `f x f` blocks and sums them with the selected dtype; divisibility, factor,
shape, and overflow expectations are not validated here.

Registration math uses NumPy. Importing `pyretinas.py` also imports local
`utils.py`; that module imports `scipy.special.erf` at module load even though
registration uses only shifting and centering helpers. SciPy is therefore a
transitive import requirement, not part of the visible registration formulas.
The files use sibling-module imports and no tracked package initializer or
Python install rule is found.

## Verification And Impact

Static verification parsed `retinas/pyretinas.py`, inventoried its class
methods, and traced both mode bindings through both composites. A no-bytecode,
in-memory probe for normal and shot-noise modes observed reference/sum object
identity, in-place reference mutation, reset to `None`/zero, and the subsequent
`TypeError`; this is `executed` evidence for that local Python environment, not
`assertion-covered` regression evidence.

The manual synthetic script is only `selected-by-test/example`: it selects
normal online mode, writes comparison output, and has no tolerance assertion.
The README offline example is `documented`; it conflicts with current reset
assignments. Shot-noise execution, input validation expectations, multi-cycle
offline intent, and native numerical parity remain unresolved.

Review this page for changes to `retinas/pyretinas.py`, imported registration
helpers in `retinas/utils.py`, demonstrated callers, or Python installation.
Then review [registration pipeline](../algorithms/registration-pipeline.md),
[parity matrix](parity-matrix.md), [public API map](../public-api-map.md),
[testing](../testing/index.md), [source map](../source-map.md), and
[change impact](../change-impact.md).
