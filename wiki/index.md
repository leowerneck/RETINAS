# RETINAS Knowledge-Base Index

## Purpose

Exhaustive inventory of live `wiki/**/*.md` pages, excluding this index itself.
Wiki is derived synthesis: reopen claim-specific repository evidence before
material claims or changes. Root instructions live in `AGENTS.md`, not here.

## Common Task Starts

| Need | Start |
| --- | --- |
| Find a term, symbol, or question | [Catalog](catalog.md) |
| Locate tracked source ownership | [Source map](source-map.md) |
| Understand component boundaries | [Architecture](architecture.md) |
| Trace a frame through registration | [Registration pipeline](algorithms/registration-pipeline.md) |
| Assess backend evidence | [Implementations](implementations/index.md) |
| Assess tests or proof strength | [Testing](testing/index.md) |
| Check generated/output safety | [Generated boundaries](generated-boundaries.md) |
| Inspect build, install, or cleanup | [Build and install](build-and-install.md) |
| Classify README/notebook evidence | [Docs and experiments](docs-and-experiments.md) |
| Map a callable/public surface | [Public API map](public-api-map.md) |
| Compare backend evidence | [Parity matrix](implementations/parity-matrix.md) |
| Investigate competing evidence | [Contradiction registry](contradictions.md) |
| Follow a change playbook | [Workflows](workflows.md) |
| Review affected pages after change | [Change impact](change-impact.md) |
| Validate KB integrity | [KB checks](lint/CHECKS.md) |

## Navigation And Operations

| Page | Kind | Purpose |
| --- | --- | --- |
| [Catalog](catalog.md) | special | Route aliases, symbols, questions, and tasks to canonical owners. |
| [Operation log](log.md) | special | Record successful material KB operations without factual/freshness authority. |
| [Workflows](workflows.md) | leaf | Route product and KB changes through owners, evidence, safety, and verification. |

## Maps And Boundaries

| Page | Kind | Purpose |
| --- | --- | --- |
| [Source map](source-map.md) | map | Inventory tracked path families and complete primary ownership. |
| [Generated boundaries](generated-boundaries.md) | leaf | Separate tracked inputs, generated products, runtime output, and destructive workflows. |
| [Architecture](architecture.md) | hub | Route layers and components without duplicating lifecycle flow. |
| [Change impact](change-impact.md) | map | Route changed path families to owners, reviewers, and verification. |
| [Public API map](public-api-map.md) | map | Separate documented, declared, defined, selected, bound, installed, executed, and asserted surfaces. |
| [Contradiction registry](contradictions.md) | map | Centralize confirmed competing evidence and smallest decisions or experiments. |
| [Build and install](build-and-install.md) | leaf | Own configuration, targets, dependencies, installation, and cleanup evidence. |
| [Docs and experiments](docs-and-experiments.md) | leaf | Classify README and notebook authority and reproducibility limits. |

## Algorithms

| Page | Kind | Purpose |
| --- | --- | --- |
| [Algorithm hub](algorithms/index.md) | hub | Define shared registration vocabulary and route algorithm questions. |
| [Registration pipeline](algorithms/registration-pipeline.md) | leaf | Own end-to-end frame and state flow across implementations. |
| [Preprocessing and correlation](algorithms/preprocessing-and-correlation.md) | leaf | Own centering, brightness, normal correlation, and shot-noise transforms. |
| [Displacement estimation](algorithms/displacement-estimation.md) | leaf | Own coordinate conventions, full-pixel selection, localized upsampling, and refinement. |
| [Reference update modes](algorithms/reference-update-modes.md) | leaf | Own first-reference, online, accumulation, reset, and continuation transitions. |

## Implementations And Testing

| Page | Kind | Purpose |
| --- | --- | --- |
| [Implementation hub](implementations/index.md) | hub | Define backend evidence vocabulary and ownership boundaries. |
| [Pure-Python implementation](implementations/python.md) | leaf | Own `Pyretinas` state, dispatch, callable surface, and lifecycle. |
| [C implementation](implementations/c.md) | leaf | Own C registration state, FFTW/BLAS boundary, selected units, and lifecycle. |
| [CUDA implementation](implementations/cuda.md) | leaf | Own CUDA host/device state, cuFFT/cuBLAS boundary, selected units, and lifecycle. |
| [Native utilities](implementations/native-utilities.md) | leaf | Own standalone C utilities and their separate build/wiring boundary. |
| [Ctypes bridge](interfaces/ctypes-bridge.md) | leaf | Own native loading, ABI bindings, array assumptions, and wrapper lifetime. |
| [Parity matrix](implementations/parity-matrix.md) | map | Compare controlled backend evidence without inferring numerical parity. |
| [Testing hub](testing/index.md) | hub | Define fixture, oracle, execution, and coverage boundaries. |
| [Synthetic comparison](testing/synthetic-comparison.md) | leaf | Own manual comparison prerequisites, side effects, outputs, and proof limits. |
| [Coverage and gaps](testing/coverage-and-gaps.md) | leaf | Own current verification strength and smallest high-value experiments. |

## Integrity

| Page | Kind | Purpose |
| --- | --- | --- |
| [KB checks](lint/CHECKS.md) | special | Specify manual and automated graph/content checks. |

Index entries use only `hub`, `leaf`, `map`, or `special`. `AGENTS.md`, this
index, product Markdown/notebooks, generated output, and temporary validator
fixtures are explicit nonentries.
