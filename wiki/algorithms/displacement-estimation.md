# Displacement Estimation

## Purpose

Own full-pixel peak selection, horizontal/vertical coordinates, signed index
wrapping, localized DFT upsampling, and sub-pixel refinement. This page records
implementation differences without claiming numerical parity.

## Ground Truth

| Layer | Exact paths and symbols |
| --- | --- |
| Pure Python | `retinas/pyretinas.py`: both `displacements_full_pixel_estimate_*`, `upsample_around_displacements`, both `displacements_sub_pixel_estimate_*` |
| C | `retinas/src/cross_correlation_c/displacements_full_pixel_estimate.c`, `upsample_around_displacements.c`, `displacements_sub_pixel_estimate.c`, both composite units |
| CUDA | normal/shot-noise `displacements_full_pixel_estimate*.cu`, `upsample_around_displacements.cu`, normal/shot-noise `displacements_sub_pixel_estimate*.cu`, `compute_kernels.cu`, composite units |
| Coordinate helper | `retinas/utils.py`: both `center_array_*_return_displacements` functions and `freq_shift` |
| Derivation/experiment | `doc/Upsampling.ipynb` |
| Caller | `retinas/synthetic_data_test.py` |

## Current Contract

### Full-Pixel Estimate And Coordinates

Normal pure Python chooses a maximum of complex modulus; CUDA computes squared
modulus and chooses its maximum. C delegates directly to
`CBLAS_IAMAX_COMPLEX` (`cblas_icamax` or `cblas_izamax`) on the complex
correlation. The Netlib BLAS
[`iamax` reference](https://www.netlib.org/lapack/explore-html/dd/d52/group__iamax.html)
selects by `|real(x)| + |imag(x)|`, so the C criterion is not equivalent to
Python/CUDA Euclidean modulus and can choose another index. Visible Python/CUDA
shot-noise paths choose a minimum of modulus or squared modulus; no C shot-noise
estimator is found. Tie behavior still needs an implementation-specific audit.

Pure Python maps NumPy `(row, column)` result to
`displacements[0]=column` and `displacements[1]=row`; C/CUDA derive the same
horizontal-first, vertical-second order from flattened `i + N_horizontal*j`
indices.

Each coordinate greater than half its corresponding dimension is reduced by
that dimension, yielding signed wrapped coordinates. Exact half-size is not
reduced because the comparison is strict `>`. Thus implementation convention
is horizontal then vertical with positive half-size retained. Physical motion
sign beyond this stored convention is unresolved without a named fixture and
oracle.

First-frame centering does not run this estimator. Composite methods instead
return helper-produced `h_0`, `v_0`, where each is the negative roll used to
center the selected extremum. Later alignment uses those displacement values in
the positive-exponent phase matrix in `freq_shift` or native reverse-shift
helpers; see [reference update modes](reference-update-modes.md).

### Localized Upsampling

For upsample factor `u`, implementations use a square local region
`S=ceil(1.5*u)` and center index `dftshift`, obtained by truncating half of `S`
toward zero. They round the current full-pixel displacement to the `1/u` grid,
reverse horizontal/vertical order where array axes require it, and compute
sample-region offsets `dftshift - displacement*u`.

Rather than zero-padding the entire correlation, runtime implementations build
Fourier kernels from signed FFT frequencies and contract them with the
conjugated image product to produce an `S x S` local sample. Pure Python uses
`fftfreq`, `exp`, and repeated `tensordot`; C uses CBLAS matrix products; CUDA
uses kernel generation and cuBLAS products.

Pure-Python composites always call localized upsampling and sub-pixel selection.
C/CUDA composites call them only when `(int)(upsample_factor + 0.5) > 1`.
Behavior for nonpositive, noninteger, or very small factors is not validated and
remains unresolved.

### Sub-Pixel Result

Pure Python selects local maximum modulus in normal mode and minimum modulus in
shot-noise mode. CUDA makes the corresponding selection over squared modulus.
C again calls its complex BLAS `iamax` selector. The chosen local offset minus
`dftshift`, divided by `u`, is added to the full-pixel estimate. Pure Python
explicitly maps local column to horizontal and row to vertical. Native flattened
matrix/index layout is defined in its source, but selector and numerical
coordinate equivalence with Python have no assertion proof.

### Notebook Boundary

`doc/Upsampling.ipynb` contains explanatory one-dimensional zero-padding and
localized-region experiments plus a kernel formula. Runtime code uses the
two-dimensional localized contraction described above; notebook cells are not
imported or registered as tests. The notebook references “the paper” without a
durable citation in the visible prose, uses experimental paths/APIs that do not
match current tracked modules, and contains stored outputs. Treat it as
`documented` derivation context, not runtime or parity proof.

NumPy's official [`fftfreq`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html)
documentation supplies upstream frequency-bin semantics. RETINAS source owns
axis ordering, offsets, selection, and update behavior.

## Verification And Impact

Static verification compared normal/shot-noise selection, flattened index
mapping, midpoint wrapping, region construction, and local-offset addition in
Python, C, and CUDA. Notebook cells were read without execution. No native
build, GPU execution, precision sweep, tie-case test, or tolerance-based
cross-backend assertion was performed.

`retinas/synthetic_data_test.py` calls full composite methods and writes all
three outputs, but contains no estimator assertion. Full-pixel tie behavior,
half-size convention intent, physical sign, and numerical parity remain
unresolved.

Review this page for estimator, kernel, phase-shift, centering, dimension, or
upsample-factor changes. Also review
[preprocessing and correlation](preprocessing-and-correlation.md),
[pure Python](../implementations/python.md),
[registration pipeline](registration-pipeline.md),
[testing](../testing/index.md), and [change impact](../change-impact.md).
