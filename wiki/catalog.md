# RETINAS Knowledge-Base Catalog

## Purpose

Route one alias, symbol, natural-language question, or task to one canonical
owner. Rows are navigation, not glossary prose.

## Routes

| Alias, symbol, question, or task | Canonical owner | Primary source/header | Documentation/test route |
| --- | --- | --- | --- |
| repository inventory | [Source map](source-map.md) | `git ls-files` and recursive Meson manifests | [Generated boundaries](generated-boundaries.md) |
| generated files | [Generated boundaries](generated-boundaries.md) | `configure`, `meson.build.in`, `retinas/utils.py` | `README.md` |
| destructive synthetic output | [Generated boundaries](generated-boundaries.md) | `retinas/utils.py` | `retinas/synthetic_data_test.py` |
| system architecture | [Architecture](architecture.md) | `retinas/meson.build`, `retinas/src/meson.build` | [Source map](source-map.md) |
| registration algorithm | [Algorithm hub](algorithms/index.md) | `retinas/pyretinas.py` and native composite functions | [Registration pipeline](algorithms/registration-pipeline.md) |
| preprocessing, brightness, shot noise | [Preprocessing and correlation](algorithms/preprocessing-and-correlation.md) | mode-specific Python/C/CUDA units | [Parity matrix](implementations/parity-matrix.md) |
| displacement sign, axes, upsampling | [Displacement estimation](algorithms/displacement-estimation.md) | estimator/upsampling units | `doc/Upsampling.ipynb` |
| offline, IIR, accumulator, reset | [Reference update modes](algorithms/reference-update-modes.md) | update/accumulation units | [Coverage and gaps](testing/coverage-and-gaps.md) |
| frame lifecycle | [Registration pipeline](algorithms/registration-pipeline.md) | `retinas/pyretinas.py`, `retinas/retinas.py` | [Testing hub](testing/index.md) |
| `compute_displacements_and_update_ref_image` | [Registration pipeline](algorithms/registration-pipeline.md) | `retinas/pyretinas.py`, `retinas/retinas.py` | `retinas/synthetic_data_test.py` |
| backend implementation | [Implementation hub](implementations/index.md) | `retinas/src/cross_correlation_c/`, `retinas/src/cross_correlation_cuda/` | [Architecture](architecture.md) |
| `Pyretinas`, NumPy implementation | [Pure-Python implementation](implementations/python.md) | `retinas/pyretinas.py` | [Reference update modes](algorithms/reference-update-modes.md) |
| C backend, FFTW, CBLAS | [C implementation](implementations/c.md) | C header, manifest, and selected units | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| CUDA backend, cuFFT, cuBLAS | [CUDA implementation](implementations/cuda.md) | CUDA headers, manifest, and selected units | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| `ctypes`, ABI, `libpath`, finalization | [Ctypes bridge](interfaces/ctypes-bridge.md) | `retinas/retinas.py` | [Public API map](public-api-map.md) |
| native utility C sources | [Native utilities](implementations/native-utilities.md) | `retinas/src/utils/*.c`, utility Makefile | [Build and install](build-and-install.md) |
| public symbol, declaration, binding, install | [Public API map](public-api-map.md) | headers, definitions, manifests, bindings | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| backend parity, precision, runtime proof | [Parity matrix](implementations/parity-matrix.md) | implementation owners | [Coverage and gaps](testing/coverage-and-gaps.md) |
| configure, Meson, install headers | [Build and install](build-and-install.md) | `configure`, Meson inputs/manifests | [Generated boundaries](generated-boundaries.md) |
| README, notebooks, stored outputs | [Docs and experiments](docs-and-experiments.md) | `README.md`, `doc/*.ipynb` | [Testing hub](testing/index.md) |
| synthetic comparison, fixed library paths | [Synthetic comparison](testing/synthetic-comparison.md) | `retinas/synthetic_data_test.py` | [Generated boundaries](generated-boundaries.md) |
| missing tests, accumulator, aliasing | [Coverage and gaps](testing/coverage-and-gaps.md) | exact implementations and runners | [Contradiction registry](contradictions.md) |
| conflicting evidence, unresolved intent | [Contradiction registry](contradictions.md) | linked competing source | [Workflows](workflows.md) |
| change playbook, ingest, query, lint, curate | [Workflows](workflows.md) | root operations and changed family | [Change impact](change-impact.md) |
| evidence strength | [Testing hub](testing/index.md) | assertions, runner, and observed execution | `retinas/synthetic_data_test.py` |
| affected documentation after change | [Change impact](change-impact.md) | changed path family | [KB checks](lint/CHECKS.md) |
| knowledge-base validation | [KB checks](lint/CHECKS.md) | `scripts/kb.py`, `scripts/tests/test_kb.py` | [Source map](source-map.md) |
| recent KB operation | [Operation log](log.md) | Git history is authoritative | inspect only relevant log entry |

When no route exists, use `rg`, reopen exact evidence, and update catalog only
if authorized work establishes a durable canonical route.
