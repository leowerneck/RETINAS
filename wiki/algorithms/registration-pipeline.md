# Registration Pipeline

## Purpose

Canonical end-to-end frame/state lifecycle from construction through
finalization. It relates pure Python, bridge, C, and CUDA static evidence plus
focused pure-Python lifecycle probes without claiming parity.

## Ground Truth

| Layer | Exact paths and important symbols |
| --- | --- |
| Pure Python | `retinas/pyretinas.py`: `Pyretinas.__init__`, mode-bound preprocessing/correlation/estimate/update functions, composite compute methods, `finalize` |
| Native bridge | `retinas/retinas.py`: `retinas.__init__`, `initialize_library_functions`, preprocessing/composite methods, `finalize` |
| C backend | `retinas/src/cross_correlation_c/retinas.h`; `state_initialize.c`, `set_first_reference_image.c`, preprocessing/correlation/estimate/upsample/update/accumulate/finalize units; C `meson.build` |
| CUDA backend | CUDA `retinas.h`, `function_prototypes.h`, state/preprocess/correlation/estimate/upsample/update/accumulate/finalize units; CUDA `meson.build` |
| Caller/example | `retinas/synthetic_data_test.py`; offline prose/example in `README.md` |

## Current Contract

Callers construct state with horizontal/vertical sizes, upsample factor, update
weights derived from optional time constant, mode, and offset. Each frame must
be preprocessed before a composite displacement method because preprocessing
populates the current image representation used by later steps. Composite
methods return horizontal then vertical displacement values.

First-frame centering occurs in pure Python or bridge preprocessing when
configured; it returns initial offsets stored as `h_0`, `v_0`. First composite
call installs a reference and returns those offsets without correlation. Later
calls correlate current/reference representations, make full-pixel estimate,
refine around it according to the implementation-specific condition described
below, then either update reference online or accumulate an aligned frame.
Averaged-reference replacement is an explicit operation. Finalization differs
by layer.

### Flow And Invariants

1. **Construction.** `Pyretinas` allocates NumPy arrays and binds mode-specific
   methods. Bridge selects C scalar types from precision, loads a supplied
   library with `CDLL`, binds requested symbols, and calls native
   `state_initialize`. C allocates FFTW-backed state/plans; CUDA allocates
   host/device state, a cuFFT plan, and cuBLAS handle.
2. **[First preprocessing](preprocessing-and-correlation.md).** Optional max/min centering is visible in pure
   Python and bridge. Normal pure Python computes `fft2(new_image)` and
   brightness. Its shot-noise path computes Fourier transforms of squared and
   reciprocal offset images. Native preprocessing accepts a `uint16` pointer;
   CUDA has separate normal/shot-noise definitions while current C source/header
   expose only normal preprocessing.
3. **First reference.** Pure Python assigns its normal or reciprocal reference.
   Bridge calls native `set_first_reference_image`. C computes only reference
   FFT there. CUDA enqueues a reference-to-sum copy and sets the host counter to
   one; completion before bridge return is not proved. These are observed
   differences, not parity conclusions.
4. **[Correlation](preprocessing-and-correlation.md).** Normal implementations form a current-frequency product
   with conjugated reference and inverse-transform it. Pure Python shot-noise
   uses squared-current against reciprocal-reference. CUDA shot-noise
   preprocessing/composite functions provide the corresponding static path;
   current C selection does not expose that symbol family.
5. **[Full-pixel estimate](displacement-estimation.md).** Normal paths use
   implementation-specific extrema criteria; C complex BLAS `iamax` is not
   equivalent to Python/CUDA modulus criteria. Visible pure-Python and CUDA
   shot-noise paths select minima. Wrapped indices above half-size are shifted
   into signed horizontal/vertical coordinates. See the linked canonical owner
   for exact selector semantics and evidence limits.
6. **[Localized refinement](displacement-estimation.md).** DFT kernels sample a region around rounded
   full-pixel displacement. Native composites call upsampling/sub-pixel stages
   only when rounded upsample factor exceeds one; pure Python composite methods
   call both directly. Max/min location adds fractional displacement divided by
   upsample factor.
7. **[Online update](reference-update-modes.md).** Current frequency image is reverse-shifted/aligned, then
   combined as `A0 * current + B1 * reference`. Shot-noise paths use reciprocal
   representation where visible.
8. **[Accumulation](reference-update-modes.md).** Aligned current representation is added and counter
   increments. Pure Python first accumulation aliases image-sum storage to its
   reference object; after averaged replacement it sets sum to `None` and
   counter to zero. C contains manifest-selected `initialize_image_sum.c`, but
   current header/bridge/call trace does not visibly invoke it before later
   addition. CUDA initializes sum during first-reference setup. These static
   findings require focused tests before behavior/support claims.
9. **[Averaged replacement](reference-update-modes.md).** C divides accumulated image by counter, copies new
   reference back into sum, and resets counter to one. CUDA divides into
   reference, zeros sum, and resets counter to zero. Pure Python divides into
   reference, then sets sum `None` and counter zero. No parity claim follows.
10. **Finalization.** Pure-Python `finalize` does nothing. Native finalizers free
    allocated state. Bridge calls native finalization only when
    `self.initialized` is true; constructor visibly initializes it false and
    does not visibly set it true after native state creation.

### Evidence Qualification

| Layer/path | Documented | Declared/defined | Build/binding | Runtime/assertion evidence |
| --- | --- | --- | --- | --- |
| Pure Python normal | README/example present | Python surface and definitions present | Imported directly by manual script | selected-by-test/example; no assertion oracle found |
| Pure Python shot noise | Docstrings/surface present | Definitions and mode binding present | Selected by constructor flag | manual script currently selects false; execution/assertion unresolved |
| C normal | README/header present | Header declarations and definitions present | 18 units build-selected; bridge requests symbols | manual script selects installed C library; execution/assertion unresolved here |
| C shot noise | Bridge has conditional requests | matching C declarations/definitions not found | not found in C selected list | unresolved/absent static evidence; no assertion proof |
| CUDA normal | README/header present | declarations and definitions present | 27 units build-selected; bridge requests symbols | manual script selects installed CUDA library; execution/assertion unresolved here |
| CUDA shot noise | Header/bridge surface present | declarations and definitions present | build-selected and bound when flag true | manual script currently selects false; execution/assertion unresolved |
| Pure-Python offline lifecycle | Not applicable | Normal and shot-noise definitions present | Selected directly by focused probe | alias/reset transitions executed locally; no regression assertion |

## Verification And Impact

Static verification must trace, for each named mode: Python/bridge caller,
declaration, definition, manifest selection, and closest example/test. Build,
load, execute, and assertion-cover only in a named environment with explicit
fixture/oracle/tolerance. Do not run synthetic generation outside an inspected
disposable output path.

Review this page for changes to pure-Python/bridge composite methods, native
state or pipeline units, headers, backend manifests, synthetic caller, and
offline README example. Use [change impact](../change-impact.md),
[implementations](../implementations/index.md), [testing](../testing/index.md),
and [generated boundaries](../generated-boundaries.md).

Stage comparisons belong to [parity matrix](../implementations/parity-matrix.md);
public callable layers belong to [public API map](../public-api-map.md).

Current gaps: focused local pure-Python probes executed normal/shot-noise
aliasing and reset transitions, but no native runtime or full product suite was
executed. Native accumulator initialization/reset, bridge finalization, C
shot-noise availability, and cross-backend numerical equivalence remain
unresolved.
