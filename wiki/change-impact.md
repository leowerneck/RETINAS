# RETINAS Change Impact Map

## Purpose And Schema

Advisory review routing for changed paths. Strict columns are `Path glob`,
`Primary owner`, `Also review`, and `Verification`. Every row matches one source
family and primary owner. `Also review` never transfers primary ownership.

| Path glob | Primary owner | Also review | Verification |
| --- | --- | --- | --- |
| `README.md` | [Docs and experiments](docs-and-experiments.md) | [Build](build-and-install.md), [Synthetic comparison](testing/synthetic-comparison.md), [Contradictions](contradictions.md) | Reconcile prose with exact implementation/build evidence |
| `configure` | [Build and install](build-and-install.md) | [Generated boundaries](generated-boundaries.md), [Workflows](workflows.md), [Contradictions](contradictions.md) | Static generation/command trace; disposable configure only if authorized |
| `meson.build.in` | [Build and install](build-and-install.md) | [Public API map](public-api-map.md), [Generated boundaries](generated-boundaries.md), [Contradictions](contradictions.md) | Trace options, dependencies, libraries, install declarations |
| `meson_options.txt` | [Build and install](build-and-install.md) | [Contradictions](contradictions.md) | Compare schema defaults with configure arguments |
| `doc/*.ipynb` | [Docs and experiments](docs-and-experiments.md) | [Displacement estimation](algorithms/displacement-estimation.md), [Testing](testing/index.md) | Static prose/cell review; do not execute or rewrite outputs by default |
| `retinas/.pylintrc` | [Workflows](workflows.md) | [Pure Python](implementations/python.md), [Testing](testing/index.md) | Run only applicable Python lint command when established |
| `retinas/meson.build` | [Build and install](build-and-install.md) | [Architecture](architecture.md), [Public API map](public-api-map.md) | Trace recursive subdirectory selection |
| `retinas/src/meson.build` | [Build and install](build-and-install.md) | [Architecture](architecture.md), [Native utilities](implementations/native-utilities.md) | Trace backend subdirectory selection and utility exclusion |
| `retinas/pyretinas.py` | [Pure Python](implementations/python.md) | [Registration pipeline](algorithms/registration-pipeline.md), [Algorithms](algorithms/index.md), [Parity](implementations/parity-matrix.md), [Testing](testing/index.md), [API map](public-api-map.md) | Static state/caller trace; proportionate Python tests |
| `retinas/retinas.py` | [Ctypes bridge](interfaces/ctypes-bridge.md) | [Registration pipeline](algorithms/registration-pipeline.md), [Parity](implementations/parity-matrix.md), [Testing](testing/index.md), [API map](public-api-map.md), [Contradictions](contradictions.md) | Trace ctypes types, symbols, lifecycle, and callers |
| `retinas/utils.py` | [Public API map](public-api-map.md) | [Generated boundaries](generated-boundaries.md), [Pure Python](implementations/python.md), [Ctypes bridge](interfaces/ctypes-bridge.md), [Algorithms](algorithms/index.md), [Synthetic comparison](testing/synthetic-comparison.md), [CUDA](implementations/cuda.md) | Route the changed callable to its canonical owner; inspect destructive output paths before synthetic generation; review exact consumers |
| `retinas/synthetic_data_test.py` | [Synthetic comparison](testing/synthetic-comparison.md) | [Generated boundaries](generated-boundaries.md), [Coverage](testing/coverage-and-gaps.md), [Parity](implementations/parity-matrix.md) | Static fixture/oracle/output audit; inspect destructive output path and use a disposable run only if authorized |
| `retinas/src/cross_correlation_c/meson.build` | [C implementation](implementations/c.md) | [Build](build-and-install.md), [API map](public-api-map.md), [Parity](implementations/parity-matrix.md) | Reconcile selected C units and header variable |
| `retinas/src/cross_correlation_c/retinas.h` | [C implementation](implementations/c.md) | [Ctypes bridge](interfaces/ctypes-bridge.md), [API map](public-api-map.md), [Generated boundaries](generated-boundaries.md), [Contradictions](contradictions.md) | Compare declarations, definitions, bridge bindings, selection |
| `retinas/src/cross_correlation_c/*.c` | [C implementation](implementations/c.md) | [Registration pipeline](algorithms/registration-pipeline.md), [Algorithms](algorithms/index.md), [Parity](implementations/parity-matrix.md), [Testing](testing/index.md) | Trace caller/state/manifest; compile/runtime only in named environment |
| `retinas/src/cross_correlation_cuda/meson.build` | [CUDA implementation](implementations/cuda.md) | [Build](build-and-install.md), [API map](public-api-map.md), [Parity](implementations/parity-matrix.md) | Reconcile selected CUDA units |
| `retinas/src/cross_correlation_cuda/retinas.h` | [CUDA implementation](implementations/cuda.md) | [Ctypes bridge](interfaces/ctypes-bridge.md), [API map](public-api-map.md), [Generated boundaries](generated-boundaries.md) | Compare state, prototype include, selection, binding |
| `retinas/src/cross_correlation_cuda/function_prototypes.h` | [CUDA implementation](implementations/cuda.md) | [Ctypes bridge](interfaces/ctypes-bridge.md), [API map](public-api-map.md) | Compare declarations, definitions, bridge bindings, selection |
| `retinas/src/cross_correlation_cuda/*.cu` | [CUDA implementation](implementations/cuda.md) | [Registration pipeline](algorithms/registration-pipeline.md), [Algorithms](algorithms/index.md), [Parity](implementations/parity-matrix.md), [Testing](testing/index.md) | Static host/device/caller trace; CUDA proof only on named host |
| `retinas/src/utils/Makefile` | [Native utilities](implementations/native-utilities.md) | [Build](build-and-install.md), [Workflows](workflows.md) | Compare utility-only object build with main Meson lists |
| `retinas/src/utils/*.c` | [Native utilities](implementations/native-utilities.md) | [Build](build-and-install.md), [Testing](testing/index.md), [Public API map](public-api-map.md) | Trace declarations/callers/build wiring before availability claims |
| `AGENTS.md` | [Root schema](../AGENTS.md) | [Wiki index](index.md), [KB checks](lint/CHECKS.md) | Run KB tests and staged/complete check as lifecycle requires |
| `wiki/**/*.md` | [Wiki index](index.md) | [KB checks](lint/CHECKS.md) | Run KB tests/check; review canonical ownership and duplicates |
| `scripts/kb.py` | [KB checks](lint/CHECKS.md) | [Source map](source-map.md) | Run tooling tests then default check; exercise explicit path and Git-range routing |
| `scripts/tests/**` | [KB checks](lint/CHECKS.md) | [Source map](source-map.md) | Run full validator test discovery and artifact check |
| `.github/workflows/*.yml` | [KB checks](lint/CHECKS.md) | [Source map](source-map.md), [Workflows](workflows.md) | Run tests/default check; inspect explicit base/head resolution and advisory conditions |

Unknown changed paths require manual source-map reconciliation; this map never
edits pages or certifies freshness.
