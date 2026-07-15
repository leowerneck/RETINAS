# Build And Install

## Purpose

Canonical owner for RETINAS configuration, native target selection,
dependencies, generated build files, installation, and cleanup. This page
separates what tracked manifests declare from what a named environment has
configured, built, installed, loaded, or executed.

## Ground Truth

| Concern | RETINAS evidence |
| --- | --- |
| User entry and generated wrappers | `configure`, `README.md` |
| Root project template and option schema | `meson.build.in`, `meson_options.txt` |
| Recursive selection | `retinas/meson.build`, `retinas/src/meson.build`, backend `meson.build` files |
| C target and public header input | `retinas/src/cross_correlation_c/meson.build`, C `retinas.h` |
| CUDA target and headers | `retinas/src/cross_correlation_cuda/meson.build`, CUDA `retinas.h`, `function_prototypes.h` |
| Generated/output safety | [Generated boundaries](generated-boundaries.md) |

Meson's official [build-option documentation](https://mesonbuild.com/Build-options.html)
defines project options in `meson_options.txt`; those declarations describe
option types and defaults. Meson's
[`install_headers()` reference](https://mesonbuild.com/Reference-manual_functions.html#install_headers)
defines header installation mechanics. RETINAS manifests remain authority for
which options and headers this repository currently supplies.

## Current Contract

### Entry Route And Generation

A clean tracked checkout has `meson.build.in` but no root `meson.build`.
`meson.build.in` still contains `@languages@`, so it is not a demonstrated
direct Meson entry point. `configure` is the evidenced front end: it checks for
the selected Meson and Ninja commands, replaces `@languages@` with selected C
and/or CUDA project languages, writes root `meson.build`, invokes `meson setup`
or `--reconfigure`, then writes a Makefile wrapper. The supplied `--ninja`
value is used only for this existence check; it is not forwarded to Meson or
the generated wrapper, so actual Ninja selection remains environment/Meson
behavior rather than a configured RETINAS choice.

`meson_options.txt` declares schema defaults of C enabled, CUDA enabled, BLAS
name `blas`, and single precision. `configure` instead defaults C to `yes` and
CUDA to `no`, then passes explicit `-Dwith-c`, `-Dwith-cuda`, `-Dblas_lib`, and
`-Dprecision` values. Schema defaults therefore do not prove an alternate
clean-checkout route.

If both backends are disabled, `configure` replaces `@languages@` with an empty
string, leaving the template's language position empty, and passes both backend
options false. No disposable configuration was run here; whether Meson accepts
that generated project is unresolved.

### Axes, Dependencies, And Targets

| Axis | Declared/configured effect | Proof limit |
| --- | --- | --- |
| C | Adds C project language and selects library target `retinas` | Source list is build-selected; no build observed here |
| CUDA | Adds CUDA project language and selects library target `retinas_cuda` | Source list is build-selected; no CUDA build/runtime observed here |
| Single precision | Defines `PRECISION=0`; C requests `fftw3f`/`fftwf` | Configuration intent only |
| Double precision | Defines `PRECISION=1`; C requests `fftw3`/`fftw` | Configuration intent only |
| BLAS name | Tries configured dependency, then `blas`, `openblas`, and `blis` | Resolution depends on environment |
| Prefix/build directory | Passes selected setup directory and optional install prefix | Generated Makefile later hard-codes `build` |

The C branch requests a C compiler, `libm`, precision-specific FFTW types and
functions, and a BLAS dependency exposing the listed single- and
double-precision CBLAS symbols. It compiles the 18 C manifest units with GNU99,
optimization, native-architecture, and precision flags. The CUDA branch
requests a CUDA compiler and Meson's `cuda >=10` dependency with cuBLAS and
cuFFT modules, then compiles the 27 CUDA manifest units with native-architecture,
optimization, and precision flags. These are manifest requirements, not proof
that the host satisfies them.

### Install, Generated Artifacts, And Cleanup

Both conditional `library()` declarations set `install: true`. Recursive source
manifests are entered unconditionally, and only the C manifest assigns
`headers`; consequently current `install_headers(headers, subdir: 'retinas')`
selects the C `retinas.h` whenever either backend is enabled. CUDA-only and
combined intended public-header surfaces remain unresolved until a
disposable-prefix install is inspected.

`configure` generates root `meson.build` and `Makefile`; Meson/Ninja generate
the chosen build tree; installation generates prefix-relative libraries and
headers. None are tracked source. Generated Makefile targets compile, test, and
install with hard-coded `-C build`; `clean` removes object files below `build`,
and `realclean` recursively removes `build`, `lib`, and `include` plus generated
root wrappers. `uninstall` removes paths derived from the generated `prefix`
variable. When `--prefix` is omitted, that variable is literally `None`, so the
wrapper does not target Meson's default install prefix; README's default-prefix
uninstall route is broken by static command tracing. Inspect paths before
cleanup or uninstall; do not run these operations merely to check Markdown.

### Evidence Stages And Unresolved Behavior

| Stage | Minimum evidence |
| --- | --- |
| Declared | Option, dependency, source, target, or install declaration in exact tracked file |
| Configured | Successful `configure`/Meson setup in named backend, precision, paths, and environment |
| Compiled/linked | Successful named target build and link artifacts in that environment |
| Installed | Disposable-prefix install inspected for exact libraries and headers |
| Loaded | Named library opened by bridge or another caller |
| Executed | Named callable path completed with stated inputs |
| Assertion-covered | Explicit oracle and tolerance passed for named mode and precision |

Current contradictions and gaps:

- `--builddir` affects Meson setup, while every generated Makefile operation
  uses `build`.
- `--default-library` is parsed with default `static` but is not added to the
  Meson command.
- `subprocess.run()` has no `check=True` or return-code branch; a returned Meson
  failure does not visibly prevent Makefile creation.
- C-only, CUDA-only, combined, and neither-backend install/configuration results
  have not been observed in disposable environments.
- Native headers warn of generated origin, but no current repository command
  establishes regeneration provenance.

Standalone [native utilities](implementations/native-utilities.md) have a
separate object-only Makefile. Main Meson selection, library linkage, install,
and supported intent are not found or unresolved as stated by that owner.

## Verification And Impact

Safe main-checkout inspection is limited to `./configure --help` and static
manifest tracing. Build/install claims require a disposable checkout and
prefix, named compilers/dependencies, selected backends and precision, exact
commands, exit results, and artifact inspection. Never promote configuration
or compilation to runtime/parity proof.

Review this page after changes to `configure`, `meson.build.in`,
`meson_options.txt`, recursive Meson manifests, native headers, README build
instructions, or generated-boundary policy. Also review
[architecture](architecture.md), [source map](source-map.md), and
[testing](testing/index.md). Header-layer evidence is summarized in
[public API map](public-api-map.md); competing build evidence is centralized in
[contradictions](contradictions.md).
