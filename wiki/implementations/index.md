# RETINAS Implementations

## Purpose

Route backend questions and define evidence vocabulary without flattening pure
Python, bridge, C, CUDA, and native utilities into one implementation.

## Read First

| Boundary | Question or use | Primary evidence |
| --- | --- | --- |
| [Pure Python](python.md) | NumPy registration state, defaults, method rebinding, lifecycle, and transitive SciPy import | `retinas/pyretinas.py` |
| [Native bridge](../interfaces/ctypes-bridge.md) | Library loading, ctypes types, requested symbols, wrapper state | `retinas/retinas.py`, FFI helper in `retinas/utils.py` |
| [C](c.md) | FFTW/CBLAS state, header, selected C units | C `retinas.h` and C `meson.build` |
| [CUDA](cuda.md) | host/device state, cuFFT/cuBLAS, variants, selected CUDA units | CUDA headers and CUDA `meson.build` |
| [Native utilities](native-utilities.md) | source-present utility functions and separate object build | `retinas/src/utils/*.c`, utility `Makefile`, main Meson manifests |
| [Parity matrix](parity-matrix.md) | controlled cross-backend comparison | implementation owners and tests |
| [Public API map](../public-api-map.md) | layered callable/install evidence | definitions, headers, manifests, bindings |
| Cross-layer flow | construction through finalization | [Registration pipeline](../algorithms/registration-pipeline.md) |

## Common Tasks

- Establish one backend fact: reopen implementation, declaration, manifest,
  caller/binding, and closest test at applicable layers.
- Assess availability: report documented/declared/defined/build-selected/bound/
  compiled/executed/assertion-covered separately.
- Add or change shared behavior: review [algorithms](../algorithms/index.md),
  [testing](../testing/index.md), and [change impact](../change-impact.md).
- Inspect generated native headers: follow
  [generated boundaries](../generated-boundaries.md).

## Ownership Boundary

This hub owns backend vocabulary and routing. It does not own backend leaf
details, ABI lifetime, cross-backend parity, public API mapping, shared
algorithm flow, build/install procedure, or test oracle semantics.

## Evidence Limits And Current Gaps

Shared names do not imply same state, default, return type, precision, mode, or
behavior. Static presence and manifest selection are established for native
source lists; build/load/runtime/assertion evidence was not produced. Main Meson
selection for native utilities is not found. Exact gaps live in backend owners,
parity matrix, and testing owners.

Parent: [wiki index](../index.md). Related: [architecture](../architecture.md)
and [catalog](../catalog.md).
