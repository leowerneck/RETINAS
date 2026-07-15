# RETINAS Source Map

KB ownership status: complete

## Purpose And Schema

Inventory in-scope tracked path families. Schema is `Source family`, `Observed
role`, `Primary owner`, and `Exact evidence`. Complete ownership requires one
linked primary owner and matching impact route per family. Presence and build
selection are separate claims. When one file contains callables with different
semantic owners, its family may route first to a strict map whose rows name the
canonical callable owners; this preserves one non-overlapping path-family route.

## Product, Build, And Documentation Inventory

| Source family | Observed role | Primary owner | Exact evidence |
| --- | --- | --- | --- |
| `README.md` | Human installation, dependency, synthetic, and offline-use prose | [Docs and experiments](docs-and-experiments.md) | `README.md` |
| `configure` | Generates root Meson/Make wrappers and invokes Meson setup | [Build and install](build-and-install.md) | `configure` |
| `meson.build.in` | Root project, dependencies, library/install declarations | [Build and install](build-and-install.md) | `meson.build.in` |
| `meson_options.txt` | Meson option schema/defaults | [Build and install](build-and-install.md) | `meson_options.txt` |
| `doc/*.ipynb` | Tracked explanatory/experimental notebooks | [Docs and experiments](docs-and-experiments.md) | `doc/Interfacing_with_the_C_library.ipynb`; `doc/Upsampling.ipynb` |
| `retinas/.pylintrc` | Python lint configuration | [Workflows](workflows.md) | `retinas/.pylintrc` |
| `retinas/meson.build` | Enters native source subtree | [Build and install](build-and-install.md) | `retinas/meson.build` |
| `retinas/src/meson.build` | Enters C and CUDA backend subdirectories | [Build and install](build-and-install.md) | `retinas/src/meson.build` |
| `retinas/pyretinas.py` | Pure-Python state and registration implementation | [Pure Python](implementations/python.md) | `retinas/pyretinas.py` |
| `retinas/retinas.py` | `ctypes` bridge to a supplied native library | [Ctypes bridge](interfaces/ctypes-bridge.md) | `retinas/retinas.py` |
| `retinas/utils.py` | Array, synthetic-data, FFI, and GPU-probe helpers | [Public API map](public-api-map.md) | `retinas/utils.py` |
| `retinas/synthetic_data_test.py` | Manual three-implementation comparison script | [Synthetic comparison](testing/synthetic-comparison.md) | `retinas/synthetic_data_test.py` |
| `retinas/src/cross_correlation_c/meson.build` | Selects 18 C sources and C header variable | [C implementation](implementations/c.md) | `retinas/src/cross_correlation_c/meson.build` |
| `retinas/src/cross_correlation_c/retinas.h` | C state and declared callable surface | [C implementation](implementations/c.md) | `retinas/src/cross_correlation_c/retinas.h` |
| `retinas/src/cross_correlation_c/*.c` | 18 tracked C implementation units, all manifest-selected | [C implementation](implementations/c.md) | `retinas/src/cross_correlation_c/meson.build` `files(...)` list |
| `retinas/src/cross_correlation_cuda/meson.build` | Selects 27 CUDA sources | [CUDA implementation](implementations/cuda.md) | `retinas/src/cross_correlation_cuda/meson.build` |
| `retinas/src/cross_correlation_cuda/retinas.h` | CUDA state, precision macros, prototype include | [CUDA implementation](implementations/cuda.md) | `retinas/src/cross_correlation_cuda/retinas.h` |
| `retinas/src/cross_correlation_cuda/function_prototypes.h` | CUDA declared callable surface | [CUDA implementation](implementations/cuda.md) | `retinas/src/cross_correlation_cuda/function_prototypes.h` |
| `retinas/src/cross_correlation_cuda/*.cu` | 27 tracked CUDA implementation units, all manifest-selected | [CUDA implementation](implementations/cuda.md) | `retinas/src/cross_correlation_cuda/meson.build` `files(...)` list |
| `retinas/src/utils/Makefile` | Separate native-utility object compilation | [Native utilities](implementations/native-utilities.md) | `retinas/src/utils/Makefile` |
| `retinas/src/utils/*.c` | Ten tracked native utility units; absent from visible main Meson source lists | [Native utilities](implementations/native-utilities.md) | `retinas/src/utils/Makefile`; recursive Meson manifests |

## Knowledge-Base Inventory

| Source family | Observed role | Primary owner | Exact evidence |
| --- | --- | --- | --- |
| `AGENTS.md` | Root schema and router | [Root schema](../AGENTS.md) | `AGENTS.md` |
| `wiki/**/*.md` | Derived knowledge graph and contracts | [Wiki index](index.md) | `wiki/index.md` exhaustive inventory |
| `scripts/kb.py` | Standard-library structural, impact, and CI-range tooling | [KB checks](lint/CHECKS.md) | `scripts/kb.py` |
| `scripts/tests/**` | Temporary-root validator, impact, resolver, and compounding fixtures | [KB checks](lint/CHECKS.md) | `scripts/tests/test_kb.py` |
| `.github/workflows/*.yml` | Advisory impact and blocking structural CI | [KB checks](lint/CHECKS.md) | `.github/workflows/kb.yml` |

## Explicit Exclusions

| Excluded family | Reason |
| --- | --- |
| `GRHayL/` | Untracked exemplar; never RETINAS authority, inventory, or impact coverage |
| `plan*.md`, `tasks?.md` | Planning coordination, not RETINAS evidence |
| `.git/` | Version-control internals, outside content ownership |
| root `meson.build`, root `Makefile` | Generated by `configure`; absent in clean tracked inventory |
| build/install trees, caches, bytecode | Generated products, not source ownership |
| synthetic images, diagnostics, results | Runtime output; destructive producer documented elsewhere |

## Reconciliation

The pre-KB product baseline recorded during bootstrap contains 74 files: four
root configuration/prose files, two notebooks, four Python files, four recursive
Meson manifests, two native headers plus one CUDA prototype header, 18 C units,
27 CUDA units, one utility Makefile, ten utility C units, and one Python lint
file. This historical count is inventory reconciliation only, never freshness
evidence. Re-run `git ls-files` and review families whenever indexed paths
change.

Related pages: [Generated boundaries](generated-boundaries.md),
[Change impact](change-impact.md), and [Architecture](architecture.md).
