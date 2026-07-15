# RETINAS Architecture

## Purpose

Route system layers, components, and boundaries. Full frame/state lifecycle is
owned only by [registration pipeline](algorithms/registration-pipeline.md).

## Read First

| Layer or boundary | Question | Primary evidence |
| --- | --- | --- |
| Public prose and experiments | What usage or derivation is documented? | `README.md`, `doc/*.ipynb` |
| Pure Python | Where are NumPy registration state/dispatch and the transitive SciPy import boundary? | [Pure Python](implementations/python.md) |
| Native bridge | How does Python load and type native symbols? | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| C backend | What C state/API/source is present and selected? | [C implementation](implementations/c.md) |
| CUDA backend | What host/device state/API/source is present and selected? | [CUDA implementation](implementations/cuda.md) |
| Native utilities | Are utility C files in main libraries? | [Native utilities](implementations/native-utilities.md) |
| Build/install | How are languages, libraries, and headers selected? | [Build and install](build-and-install.md) |
| Generated/runtime output | What may be overwritten or deleted? | [Generated boundaries](generated-boundaries.md) |
| Evidence and verification | What does a check actually prove? | [Testing](testing/index.md) |

## Common Tasks

- Locate path ownership in [source map](source-map.md).
- Trace construction through finalization in
  [registration pipeline](algorithms/registration-pipeline.md).
- Compare evidence layers without assuming parity via
  [implementation hub](implementations/index.md) and
  [parity matrix](implementations/parity-matrix.md).
- Trace callable/install evidence in [public API map](public-api-map.md).
- Review affected owners/checks through [change impact](change-impact.md).

## Ownership Boundary

This page owns component routing and boundaries only. Algorithm state
transitions belong to registration pipeline; backend-specific state belongs to
implementation owners; ABI lifetime belongs to bridge owner; build commands
and install surface belong to build owner; fixture/oracle semantics belong to
testing owners.

## Evidence Limits And Current Gaps

File presence, declaration, manifest selection, binding, build, load, execution,
and assertion coverage remain distinct. Recursive Meson manifests select C and
CUDA source lists, but native utilities are absent from visible main lists.
No product build/runtime was performed for this routing page.

Parent: [wiki index](index.md). Related: [catalog](catalog.md),
[algorithms](algorithms/index.md), and [generated boundaries](generated-boundaries.md).
