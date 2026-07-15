# RETINAS Parity Matrix

## Purpose And Schema

Compare implementation evidence without promoting shared names to numerical
parity. Strict columns are `Concern`, `Pure Python`, `C`, `CUDA`, `Bridge`,
`Test evidence`, and `Parity result`. Cells use controlled evidence terms and
explicit unknown states; no blank means support.

## Evidence Matrix

| Concern | Pure Python | C | CUDA | Bridge | Test evidence | Parity result |
| --- | --- | --- | --- | --- | --- | --- |
| Normal preprocessing/correlation | Defined; NumPy FFT path | Declared, defined, build-selected | Declared, defined, build-selected | Bound in normal mode | Selected-by-test/example; no oracle | Unresolved |
| Shot-noise preprocessing/correlation | Defined and constructor-selected | Requested state fields exist, but matching callable family absent/not found | Defined and build-selected | Requests mode-specific family | Driver selects false; no assertion | Unresolved; C availability not established |
| First-image centering | Defined in shared Python helper and used by Python/bridge | Not applicable inside backend | Not applicable inside backend | Uses same helper before native pointer call | Driver selects max; no centering assertion | Same helper is visible, but native end-to-end result unresolved |
| Horizontal/vertical result order and signed wrapping | Defined horizontal-first in estimator | Defined horizontal-first | Defined horizontal-first | Returns two-element native result | No coordinate/sign oracle | Unresolved numerical/physical-sign parity |
| Precision | NumPy transform state is complex128; outputs float64 after first result | Single/double selected by compile-time macro | Single/double selected by compile-time macro | Caller chooses single/double ABI without artifact probe | Driver selects native single only | Unresolved for every cross-backend precision pair |
| Online/IIR update | Defined; missing coefficients when time constant is `None` | Declared, defined, build-selected | Declared, defined, build-selected | Bound | Driver selects normal online mode; no assertion | Unresolved |
| First offline accumulation | Defined but sum aliases reference | Allocated sum/counter are not initialized in visible bridge path | First-reference enqueues sum copy and sets host counter one; completion before return unresolved | First call sets reference and returns | No offline native assertion | Unresolved; state transitions differ |
| Later offline accumulation | Defined with in-place `+=` | Defined/build-selected against unresolved initial state | Defined/build-selected from initialized sum | Bound | No state oracle | Unresolved |
| Averaged replacement and continuation | Defined; resets sum to `None`, counter zero; next addition executed as `TypeError` | Defined; copies average to reference and sum, counter one | Defined; zeroes sum, counter zero | Bound | Python probe executed; no regression assertion; native unresolved | Not parity-confirmed; visible reset contracts differ |
| Reference/sum aliasing | Executed object identity after first offline frame; later `+=` mutates reference | Separate allocations; initialization chain unresolved | Separate allocations; explicit copy | Not directly observable from wrapper | No approved alias contract assertion | Not parity-confirmed |
| Reference retrieval | No dedicated getter; attributes are directly accessible | Time/frequency getters declared, defined, build-selected, unbound | One getter declared, defined, build-selected, unbound | Getter bindings not found | No retrieval test | Unresolved public/retrieval parity |
| GPU probe | Not applicable | Not applicable | `gpu_works` defined/build-selected but undeclared | Separate utility binding | No probe run recorded | Not applicable across implementations; CUDA execution unresolved |
| 4x4 rebinning | Python helper `rebin` is defined | Rebin preprocessing declared, defined, build-selected, unbound | Matching rebin path not found | Binding not found | No rebin assertion | Unresolved; surfaces differ |
| Finalization/cleanup | `finalize` is no-op | Finalizer declared, defined, build-selected | Finalizer declared, defined, build-selected | Guard remains false after state creation, so visible wrapper path does not call native finalizer | No lifecycle assertion | Not parity-confirmed |
| Input boundary | No dtype/shape validation | Reads configured count of `uint16_t` | Reads configured count of `uint16_t` | Casts raw NumPy address without shape/contiguity checks | Driver supplies matching binary images; no negative tests | Unresolved safety/validation parity |
| Numerical displacement | Max/min and localized DFT defined | Normal max path defined; BLAS selector semantics differ | Normal/shot-noise squared-magnitude selectors defined | Dispatches selected native composite | Results written only; no tolerance | No parity-confirmed combination |

## Canonical Routes

- Stage formulas: [preprocessing/correlation](../algorithms/preprocessing-and-correlation.md),
  [displacement estimation](../algorithms/displacement-estimation.md), and
  [reference updates](../algorithms/reference-update-modes.md).
- State/API evidence: [pure Python](python.md), [C](c.md), [CUDA](cuda.md),
  [native utilities](native-utilities.md), [bridge](../interfaces/ctypes-bridge.md),
  and [public API map](../public-api-map.md).
- Proof requirements: [coverage and gaps](../testing/coverage-and-gaps.md) and
  [synthetic comparison](../testing/synthetic-comparison.md).

## Verification And Impact

Reopen exact source for every changed cell. `Parity-confirmed` requires named
implementations, backend, mode, precision, centering, fixture, tolerance/oracle,
environment, and observed assertion. Current matrix has no parity-confirmed
row. Changes to any implementation, bridge dispatch, algorithm contract, or
test oracle require review here without transferring primary ownership.
