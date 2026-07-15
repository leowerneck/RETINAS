# Native Utility Sources

## Purpose

Canonical owner for standalone C utilities under `retinas/src/utils`, their
separate object-only Makefile, declarations/callers found or missing, and main
build/install boundary. These sources are not the C registration backend.

## Ground Truth

| Evidence | Exact paths |
| --- | --- |
| Utility definitions | `retinas/src/utils/*.c` |
| Separate build route | `retinas/src/utils/Makefile` |
| Main build selection | `retinas/src/meson.build`, `meson.build.in` |
| Nearby backend declarations | `retinas/src/cross_correlation_c/retinas.h` |

## Current Contract

Ten tracked C files define extrema helpers for real, complex, and `uint16_t`
arrays; circular rolling; scalar/SIMD complex products; and Poisson/gamma random
helpers. Several include `image_analysis.h`, but no tracked file with that name
is found. `poisson.c` calls `gsl_ran_binomial` without a visible local
declaration or a GSL include. These are static source observations, not compile
results or diagnoses.

The local Makefile expands every `.c` file, compiling each to an object with
GCC-oriented flags. It defines only object compilation and cleanup: no library,
header installation, executable, or test target. Main recursive Meson files
enter only `cross_correlation_c` and `cross_correlation_cuda`; no reference to
the utility directory or its objects is found in those manifests.

No tracked common utility header, main-backend caller, Python binding, README
usage, or install declaration is found. Therefore these files are `defined` and
selected only by their separate Makefile; main-library `build-selected`,
`bound`, `installed`, `executed`, and `assertion-covered` states are unresolved
or not found as stated. Maintained, experimental, or dormant intent is also
unresolved.

## Verification And Impact

Before claiming availability, reconcile each definition with a tracked header,
caller, selected build target, linked artifact, and test. A disposable compile
may establish only its named environment and must handle missing headers and
dependencies explicitly. Review this page for changes to utility sources,
their Makefile, recursive Meson manifests, or any new binding; also review
[build and install](../build-and-install.md), [public API map](../public-api-map.md),
and [change impact](../change-impact.md).
