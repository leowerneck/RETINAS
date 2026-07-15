# Preprocessing And Correlation

## Purpose

Own first-frame centering, brightness, normal correlation, and shot-noise
transforms once. Backend allocation and API availability remain implementation
concerns; shared names do not establish parity.

## Ground Truth

| Layer | Exact paths and symbols |
| --- | --- |
| Pure Python | `retinas/pyretinas.py`: both `preprocess_new_image_and_compute_brightness_*`, both `set_first_reference_image_*`, both `cross_correlate_ref_and_new_images_*` |
| Centering helpers | `retinas/utils.py`: `center_array_max_return_displacements`, `center_array_min_return_displacements` |
| Bridge | `retinas/retinas.py`: `preprocess_new_image_and_compute_brightness`, `initialize_library_functions` |
| C normal path | `retinas/src/cross_correlation_c/typecast_input_image_and_compute_brightness.c`, `set_first_reference_image.c`, `cross_correlate_ref_and_new_images.c`, `meson.build` |
| CUDA normal/shot-noise paths | `retinas/src/cross_correlation_cuda/typecast_input_image_and_compute_brightness.cu`, `typecast_input_image_and_compute_brightness_shot_noise.cu`, `set_first_reference_image.cu`, `cross_correlate_ref_and_new_images.cu`, both shot-noise composite units, `meson.build` |
| Caller | `retinas/synthetic_data_test.py` |

## Current Contract

### Centering And Brightness

Centering is a Python-side first-frame operation in both `Pyretinas` and the
native bridge. With `center_first_image='max'` or `'min'`, helper code finds the
selected flattened extremum, computes row/column rolls toward array center,
returns a rolled array, and records `h_0=-column_roll`, `v_0=-row_roll`.
`None` disables it. Type and value checks occur only while `first_image` is
true. Centering uses `numpy.roll`, so it is circular rather than padded.

Brightness excludes shot-noise offset and transformed values. Pure Python sums
the image with `dtype=float64`; C loops over typecast `uint16` values; CUDA
typecasts then uses its cuBLAS absolute-sum call. Rolling preserves the set of
pixel values, but no cross-backend precision/equality claim follows.

### Normal Correlation

Mathematical data flow visible in all normal implementations is:

```text
F_new = FFT2(new image)
P = F_new * conjugate(F_reference)
C = inverse FFT2(P)
```

Pure Python computes `F_new` during preprocessing, stores `P` in
`image_product`, and stores `C` in `cross_correlation`. C and CUDA preprocessing
store typecast time-domain data; their correlation functions compute the
forward transform before product and inverse transform. First-reference setup
transforms the first preprocessed image. Library normalization and precision
must be considered before comparing correlation values.

### Shot-Noise Transform

For input `I` and configured offset `o`, visible pure-Python and CUDA paths form
the squared current representation `(I+o)^2` and reciprocal representation
`1/(I+o)`. First reference uses the reciprocal transform. Correlation uses:

```text
P_shot = FFT2((I_new + o)^2) * conjugate(FFT2(1/(I_reference + o)))
C_shot = inverse FFT2(P_shot)
```

Pure Python materializes both transforms in preprocessing. CUDA stores squared
and reciprocal time-domain arrays, transforms the squared array in the common
correlation function, and transforms reciprocal current before online update or
sum addition in shot-noise composites. Offset validity is not checked; zero in
`I+o` can therefore reach reciprocal calculation.

Current C manifest/header/source expose the normal preprocessing/correlation
family but no matching shot-noise preprocessing or composite definitions were
found. Bridge conditional symbol requests and CUDA definitions do not establish
C availability.

### Formula, Implementation, And Evidence

| Contract | Formula evidence | Defined/build-selected evidence | Runtime/test evidence |
| --- | --- | --- | --- |
| Center first image | Helper arithmetic in `retinas/utils.py` | Python and bridge call helpers | manual script selects `'max'`; no centering assertion |
| Normal transform/correlation | Product expressions in Python, C, CUDA | all three definitions; native units manifest-selected | manual script selects normal mode; output only, no oracle assertion |
| Shot-noise transform/correlation | Python and CUDA expressions above | Python defined; CUDA units defined/build-selected; C not found | manual script sets `shot_noise=False`; execution unresolved |
| Brightness | Sum implementations above | Python/C/CUDA defined; native units build-selected | values computed by manual script but not asserted |

NumPy's official [FFT routines](https://numpy.org/doc/stable/reference/routines.fft.html)
define upstream transform conventions only. Repository source remains authority
for which arrays RETINAS transforms and when.

## Verification And Impact

Static verification traces preprocessing to first-reference and correlation
consumers, then checks native manifest selection. No native build, CUDA run, or
numerical comparison was performed. `retinas/synthetic_data_test.py` is
`selected-by-test/example`, normal-only in its current constants, and lacks
assertions or tolerances.

Review this page when preprocessing, centering helpers, mode dispatch, native
transform units, or offset handling changes. Also review
[pure Python](../implementations/python.md),
[displacement estimation](displacement-estimation.md),
[registration pipeline](registration-pipeline.md),
[testing](../testing/index.md), and [change impact](../change-impact.md).
