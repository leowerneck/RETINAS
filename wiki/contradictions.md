# RETINAS Contradiction Registry

## Purpose And Schema

Centralize only confirmed competing evidence. Strict columns are `ID`,
`Bounded proposition`, `Competing evidence`, `Safe wording`, `Impact and
owner`, and `Smallest decision or experiment`. Rows do not infer desired
behavior or severity.

## Confirmed Contradictions

| ID | Bounded proposition | Competing evidence | Safe wording | Impact and owner | Smallest decision or experiment |
| --- | --- | --- | --- | --- | --- |
| C01 | Bridge shot-noise default | `retinas.retinas` signature defaults true; its constructor docstring says false; `Pyretinas` defaults false; synthetic driver explicitly selects false | Current bridge signature selects shot noise by default; prose and sibling/default usage disagree | Construction and C symbol loading; [bridge](interfaces/ctypes-bridge.md) | Maintainer chooses bridge default, then one constructor test checks selected symbols |
| C02 | Bridge native lifetime | Constructor creates native state while `initialized` remains false; `finalize` and `__del__` free only when true | Visible wrapper lifecycle does not reach native finalization after successful construction; runtime resource impact and intended lifecycle unresolved | C/CUDA cleanup; [bridge](interfaces/ctypes-bridge.md) | Instrument matching native build and assert one finalizer call after explicit finalize and destruction |
| C03 | Offline availability | README says offline algorithm is Python-only; C/CUDA sources are defined/build-selected and bridge binds offline composites | Native offline surface exists statically, but support and working behavior are unresolved | Public surface and accumulator state; [docs](docs-and-experiments.md), [API map](public-api-map.md) | Maintainer states supported backends; then backend-specific offline tests validate approved set |
| C04 | Pure-Python fixed reference | README says accumulation keeps reference fixed; first sum aliases reference and later `+=` mutates shared object | Current pure-Python offline implementation does not preserve independent first-reference storage | Offline numerical meaning; [reference updates](algorithms/reference-update-modes.md) | Maintainer defines alias contract; object-identity and three-frame oracle test it |
| C05 | Pure-Python averaged reset | README says sum becomes current reference and counter one; code sets sum to `None` and counter zero without restoring first-image state | Current implementation and documented continuation disagree; next accumulation executed as `TypeError` locally | Multi-cycle offline use; [pure Python](implementations/python.md) | Define empty-versus-seeded next cycle, then test update plus next frame in both modes |
| C06 | C offline initialization | Selected `initialize_image_sum.c` seeds sum/counter; bridge/header/caller trace does not invoke it, while allocation and first-reference setup do not initialize those fields | Initializer exists but visible bridge offline flow reaches addition without it; runtime outcome unresolved | C offline state; [C implementation](implementations/c.md) | Named-precision memory-instrumented first/second-frame test with sum/counter oracle |
| C07 | Build directory | `configure --builddir` passes chosen setup path; generated Makefile always operates on `build` | Custom setup directory and generated wrapper commands disagree | Build, test, install, and cleanup paths; [build](build-and-install.md) | Configure disposable custom directory and inspect/run only non-destructive wrapper target |
| C08 | CUDA option default | `meson_options.txt` defaults CUDA true; `configure` defaults CUDA no and passes an explicit false value | Option-schema and evidenced front-end defaults differ; neither implies an alternate clean-checkout route | Configuration expectations; [build](build-and-install.md) | Maintainer documents canonical entry/default; test generated Meson arguments |
| C09 | Installed native headers | Root install rule uses `headers`; only C manifest assigns it, even when CUDA-only is selected, while CUDA owns separate headers | Declared install surface selects C header; intended CUDA-only/combined public headers unresolved | Native public API; [build](build-and-install.md), [API map](public-api-map.md) | Inspect disposable C-only/CUDA-only/combined prefixes, then decide intended header set |
| C10 | Pure-Python offset default | `Pyretinas` signature defaults `offset=-1`; its constructor docstring says `default=0` | Current signature selects `-1`; documented default disagrees, and intended shot-noise offset remains unresolved | Shot-noise preprocessing and reciprocal validity; [pure Python](implementations/python.md) | Maintainer chooses default, then constructor/preprocessing tests cover zero-denominator handling |

## Registry Policy

Owners hold detailed narrative; this map holds bounded propositions. Remove or
rewrite a row only after current evidence eliminates competition or a maintainer
decision plus verification resolves it. Mere uncertainty without competing
evidence belongs in owner gaps, not here.

## Verification And Impact

For each row reopen both evidence sides and linked owner. Resolution requires
the smallest named decision/experiment in scope, owner updates, this registry,
applicable tests, impact routes, and one material operation-log entry. Product
changes require separate authorization.
