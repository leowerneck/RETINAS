# RETINAS Public API Map

## Purpose And Schema

Map callable and installed surfaces by evidence layer without deciding which
ones maintainers support. Strict columns are `Surface`, `Documented`,
`Declared`, `Defined`, `Build-selected`, `Bound`, `Installed`, `Executed`,
`Asserted`, and `Canonical owner`. Every cell names an observed layer or an
explicit `absent`, `not found`, `not applicable`, or `unresolved` state. This
map is also the module-family router for heterogeneous `retinas/utils.py`;
behavior remains owned by each row's canonical owner.

## Evidence Rules

Python class attributes are defined surfaces, not automatically supported
public APIs. Native headers establish declarations; source units establish
definitions; manifests establish build selection; `ctypes` setup establishes
bindings. An install declaration is not an observed install. Support intent
remains unresolved until maintainers define it.

## Layered Surface Map

| Surface | Documented | Declared | Defined | Build-selected | Bound | Installed | Executed | Asserted | Canonical owner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Pyretinas` constructor and high-level online composite | README uses constructor; synthetic driver selects online path | Not applicable: Python has no separate header | Defined as class and methods in `retinas/pyretinas.py` | Not applicable: no tracked Python install target found | Not applicable: direct Python calls | Not found in tracked install declarations | Selected-by-test/example; no fresh online run recorded | Absent: driver has no oracle | [Pure Python](implementations/python.md) |
| `Pyretinas` offline composite and averaged replacement | README documents both | Not applicable | Defined in `retinas/pyretinas.py` | Not applicable | Not applicable | Not found | Executed only by focused local lifecycle probe described by owner | Absent: no regression assertion | [Pure Python](implementations/python.md) |
| `Pyretinas` stage methods | Docstrings document many callables | Not applicable | Defined in `retinas/pyretinas.py` | Not applicable | Not applicable: direct Python calls | Not found | Selected indirectly by examples; complete execution matrix unresolved | Absent | [Pure Python](implementations/python.md) |
| `center_array_max_return_displacements`, `center_array_min_return_displacements` | Docstrings document callables | Not applicable | Defined in `retinas/utils.py` | Not applicable | Not applicable: direct Python calls | Not found | Max selected by examples; min execution unresolved | Absent | [Preprocessing and correlation](algorithms/preprocessing-and-correlation.md) |
| `freq_shift` | Docstring documents callable | Not applicable | Defined in `retinas/utils.py` | Not applicable | Not applicable: direct Python call | Not found | Selected by pure-Python update examples; fresh execution unresolved | Absent | [Reference update modes](algorithms/reference-update-modes.md) |
| `Gaussian_image`, `Poisson_image`, `generate_synthetic_image_data_set` | Docstrings document callables; README selects generator | Not applicable | Defined in `retinas/utils.py` | Not applicable | Not applicable: direct Python calls | Not found | Selected by manual synthetic comparison; fresh execution unresolved | Absent | [Synthetic comparison](testing/synthetic-comparison.md) |
| `rebin` | Docstring documents callable | Not applicable | Defined in `retinas/utils.py` | Not applicable | Not applicable: direct Python call | Not found | No current caller or execution found | Absent | [Pure Python](implementations/python.md) |
| `setup_library_function` | Docstring documents callable | Not applicable | Defined in `retinas/utils.py` | Not applicable | Assigns native `argtypes` and `restype` during bridge initialization | Not found | Selected by bridge construction; successful load unresolved | Absent | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| `gpu_works` | Docstring documents callable | Not applicable | Defined in `retinas/utils.py` | Not applicable | Binds native `gpu_works` when called | Not found | No probe run recorded | Absent | [CUDA implementation](implementations/cuda.md) |
| Bridge constructor and composite methods | README examples use bridge indirectly through synthetic driver | Not applicable | Defined in `retinas/retinas.py` | Not applicable | Bound to caller-supplied library symbols | Not found as Python package | Selected-by-test/example; successful library load unresolved | Absent | [Ctypes bridge](interfaces/ctypes-bridge.md) |
| C normal bridge family | README documents C implementation | Declared in C `retinas.h` | Defined in selected C units | Build-selected when `with-c` is true | Bound by bridge in normal mode | Library and C header install are declared; observed install unresolved | Driver selects fixed installed path; execution unresolved | Absent | [C implementation](implementations/c.md) |
| C shot-noise bridge family | Bridge flag and docstrings expose selection | Absent from C header | Not found in C definitions | Absent from C manifest | Bridge requests five names when enabled | Unresolved because requested symbols are not selected | Unresolved; bridge initialization cannot be inferred | Absent | [C implementation](implementations/c.md) |
| C rebin, reverse-shift, and reference getters | Not found in README | Declared in C `retinas.h` | Defined in selected C units | Build-selected | Not found in bridge bindings | C header install declared; library export/install unobserved | Unresolved | Absent | [C implementation](implementations/c.md) |
| C `initialize_image_sum` | Not found | Absent from C header | Defined | Build-selected | Not found; no repository caller found | Library install declared; symbol availability unobserved | Unresolved | Absent | [C implementation](implementations/c.md) |
| CUDA normal bridge family | README documents CUDA implementation | Most stages declared in prototype header; offline composites omitted | Defined in selected CUDA units | Build-selected when `with-cuda` is true | Bound by bridge | CUDA library install declared; CUDA header install not selected | Driver selects fixed installed path; execution unresolved | Absent | [CUDA implementation](implementations/cuda.md) |
| CUDA shot-noise bridge family | Bridge flag/docstrings expose selection | Stage declarations exist, but shot-noise online and both offline composites are omitted | Defined in selected CUDA units | Build-selected | Bound by bridge when enabled | CUDA library install declared; header install unresolved | Driver selects normal mode only; execution unresolved | Absent | [CUDA implementation](implementations/cuda.md) |
| CUDA reference getter and diagnostic printers | Not found in README | Declared where named in prototype header | Defined in selected CUDA units | Build-selected | Not found in bridge | Library install declared; observed exports/install unresolved | Unresolved | Absent | [CUDA implementation](implementations/cuda.md) |
| CUDA `gpu_works` probe | Utility wrapper exposes probe | Absent from CUDA prototype header | Defined in selected CUDA unit | Build-selected | Bound separately by `retinas/utils.py` | CUDA library install declared; observed install unresolved | No probe run recorded | Absent | [CUDA implementation](implementations/cuda.md) |
| Native headers | README implies installed development surface | C and CUDA headers contain declarations | Not applicable | C header assigned to install variable; CUDA headers are not | Not applicable | C header install declared when either backend enabled; observed CUDA-only/combined result unresolved | Not applicable | Not applicable | [Build and install](build-and-install.md) |
| Standalone native utilities | Not found in README | No tracked common utility header found | Defined in ten C files | Separate Makefile selects object compilation; main Meson selection not found | Not found | Not found in main install declarations | Unresolved | Absent | [Native utilities](implementations/native-utilities.md) |

## Verification And Impact

For a new native symbol, reconcile declaration, definition, backend manifest,
bridge signature, install declaration and observed prefix, caller, and explicit
test oracle. For Python, reconcile definition, import/package installation,
documented caller, execution, and assertions. Review this map after changes to
headers, Python surfaces, manifests, install rules, or bindings; primary source
behavior remains with the linked algorithm, implementation, bridge, build, or
testing owner.
