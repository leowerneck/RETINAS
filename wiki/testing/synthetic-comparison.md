# Synthetic Comparison

## Purpose

Canonical owner for `retinas/synthetic_data_test.py`: prerequisites, fixture
generation, fixed paths, cost, side effects, outputs, cleanup, registration,
and proof limits. This is a manual three-implementation comparison, not an
assertion-based parity test.

## Ground Truth

| Concern | Exact evidence |
| --- | --- |
| Documented invocation and outputs | `README.md` synthetic-data section |
| Driver, parameters, library paths, processing, writes | `retinas/synthetic_data_test.py` |
| Generator, random inputs, recursive deletion, image writes | `retinas/utils.py`: `generate_synthetic_image_data_set`, `Poisson_image` |
| Native loading and calls | `retinas/retinas.py` |
| Pure-Python calls | `retinas/pyretinas.py` |
| Build and test registration | `configure`, `meson.build.in`, recursive `meson.build` files |
| Output safety | [Generated boundaries](../generated-boundaries.md) |

Meson's official [unit-test documentation](https://mesonbuild.com/Unit-tests.html)
uses `test()` to register tests and `meson test` to run them. RETINAS manifests
determine whether this repository currently registers the synthetic driver.

## Current Contract

### Prerequisites And Fixed Inputs

The README route configures both native backends with repository-local install
prefix, builds and installs them, changes into `retinas/`, then invokes
`python synthetic_data_test.py`. Static prerequisites are:

- Python with NumPy and SciPy for the driver, pure-Python implementation, and
  generator;
- configured, compiled, linked, and installed C and CUDA libraries plus their
  Meson dependencies;
- a working CUDA runtime/device for CUDA execution; and
- execution from the documented `retinas/` working directory because imports
  are bare module names and library/output paths are relative.

The driver hard-codes `../lib/x86_64-linux-gnu/libretinas.so` and
`../lib/x86_64-linux-gnu/libretinas_cuda.so`. It therefore assumes that exact
Linux-style install layout. It instantiates C, CUDA, and Python together using
single precision for native bridges, normal rather than shot-noise mode,
first-image max centering, a 256 by 128 image, upsample factor 256, time constant
10, and 1,000 generated moves after the initial frame.

The driver passes `offset=5.76`, but the generator uses that value only in
output names: both `Poisson_image` calls omit `offset`, so generated pixels use
the helper's default zero offset. Normal-mode `Pyretinas` does not consume its
stored offset, and the normal-mode bridge replaces the supplied value with
`-1`. Thus `images_w8_o5.76` is a label, not evidence of a 5.76 image
background or an active normal-mode algorithm parameter.

Random displacement and Poisson-noise generation has no visible seed. A run
creates and processes 1,001 binary images across all three implementations;
the README describes minutes of runtime. This is a high-cost, nondeterministic
manual workflow, not a small Markdown or CI check.

### Side Effects And Cleanup Boundary

With the documented working directory, `outdir = "out"` resolves to
`retinas/out`. Before generating any fixture, the utility calls
`rmtree(outdir, ignore_errors=True)`, then recreates it. An existing selected
directory and all contents can be lost. Never run this generator in the main
checkout or against an unreviewed path.

The workflow then writes:

- `retinas/out/images_w8_o5.76/image_*.bin`: 1,001 `uint16` image files;
- `retinas/out/displacements_w8_o5.76.txt`: generated analytic moves;
- `retinas/out/results.txt`: C, CUDA, and Python numerical displacement columns;
  and
- `retinas/diagnostics.txt`: formatted per-implementation displacement rows.

Files persist after completion. A safe future experiment must use a disposable
checkout/output location, record its resolved paths before execution, and
remove only that inspected disposable output afterward. Generated images,
diagnostics, and results are never KB or tracked source.

### Execution And Proof Ladder

| Stage | What current evidence establishes | Missing proof |
| --- | --- | --- |
| Documented | README gives a Linux command sequence and expected files | No current run |
| Configured | Driver assumes both backends and local prefix | No observed setup in a named environment |
| Build-selected | Manifests select C and CUDA targets and sources | No observed compile/link |
| Load-selected | Driver constructs bridge objects for both fixed `.so` paths | No observed successful `CDLL` load |
| Run-selected | Loop calls preprocessing and online reference update for C, CUDA, and Python | No observed completion here |
| Result-producing | Source writes diagnostics and numerical columns | Files alone would not compare values |
| Assertion-covered | No `assert`, tolerance, pass/fail comparison, or analytic-result check is present | Explicit oracle absent |
| Parity-confirmed | Not established | Named fixture, seed, backend/mode/precision, tolerance, and successful assertion run required |

Brightness values are computed for each implementation but not compared. The
analytic moves and numerical results are written to separate files; the driver
does not evaluate their error or compare implementations. Successful process
exit would prove only that selected paths completed for that run.

No `test()` call is present in current tracked Meson manifests, so the driver
is not Meson-registered. Generated Makefile `check` delegates to `meson test`,
but that wrapper does not create registrations. No pytest/unittest registration
or product-test CI job selects this driver; tracked `.github/workflows/kb.yml`
validates KB structure/tooling only.

## Verification And Impact

Default verification is static: trace README invocation, resolved relative
paths, generator deletion, imports/dependencies, fixed parameters, every write,
three implementation calls, finalization, assertions, and recursive Meson
registration. Do not execute notebooks or synthetic generation as verification.

Any authorized runtime proof needs a disposable checkout/prefix, reviewed
output paths, exact tool/library/device environment, deterministic fixture or
recorded seed, selected backend/mode/precision, explicit numeric oracle and
tolerance, cleanup, and captured exit/assertion result. Review this page after
changes to the driver, generator, README instructions, library layout, bridge,
backend manifests, or test registration. Also review
[coverage and gaps](coverage-and-gaps.md), [build and install](../build-and-install.md),
and [registration pipeline](../algorithms/registration-pipeline.md).
