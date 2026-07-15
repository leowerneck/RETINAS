# Generated And Destructive Boundaries

## Purpose

Separate tracked inputs from generated configuration, build/install products,
runtime output, and destructive workflows. This page owns safe handling, not
build support claims.

## Ground Truth

| Boundary | Exact repository evidence | Observed status |
| --- | --- | --- |
| Configuration inputs | `configure`, `meson.build.in`, `meson_options.txt` | Tracked, manually editable source |
| Recursive manifests | `retinas/meson.build`, `retinas/src/meson.build`, backend `meson.build` files | Tracked, manually editable source |
| Root wrappers | `configure` writes root `meson.build` and `Makefile` | Generated and untracked in clean checkout |
| Build/install products | generated Makefile uses `build`, `lib`, and `include` paths | Generated/runtime products |
| Synthetic outputs | `retinas/synthetic_data_test.py`, `retinas/utils.py` | Runtime images, displacements, results, diagnostics |
| Native headers | both backend `retinas.h` files carry generated-origin warnings | Tracked; canonical regeneration command not found |

## Current Contract

`configure` substitutes selected languages into `meson.build.in`, writes root
`meson.build`, invokes Meson setup/reconfigure, then writes a Makefile wrapper.
The root outputs are not tracked source. Generated Makefile commands visibly use
`build` even though `configure --builddir` can pass another setup path; treat
that as observed mismatch, not permission to edit generated output.

Build and install trees may contain compiled libraries, objects, Meson/Ninja
state, and installed headers. They are evidence only when created in a named,
disposable environment. Never commit them as KB content.

`generate_synthetic_image_data_set(outdir, ...)` calls
`rmtree(outdir, ignore_errors=True)` before recreating it. Inspect and approve
the exact output path; use a disposable directory. The manual synthetic script
uses `out/` and writes `diagnostics.txt` plus result/displacement/image files.

Native header comments say not to edit by hand, but tracked RETINAS evidence
does not reveal a canonical regeneration command. Preserve them and report
provenance as `unresolved`; do not guess a generator.

### Expected Generated Outputs

Only this table exempts expected-but-absent repository paths from authority-path
existence checks.

| Expected path | Producer | Safe handling |
| --- | --- | --- |
| `meson.build` | `configure` from `meson.build.in` | Generate only for authorized build work; do not edit as source |
| `Makefile` | `configure` | Generate only for authorized build work; commands may remove files |
| `build/**` | Meson/Ninja and generated Makefile | Use disposable build directory; do not commit |
| `lib/**`, `include/**` | local install/cleanup workflow | Use disposable prefix; inspect before uninstall/cleanup |
| `retinas/out/**`, `retinas/diagnostics.txt` | manual synthetic comparison | Output directory is deleted/recreated; do not run in user data path |
| `__pycache__/**`, `*.pyc`, `*.pyo` | Python runtime | Remove/avoid; never source evidence |

## Verification And Impact

- Before configuration: `git status --short`; confirm root outputs and selected
  build/prefix paths are disposable.
- Before synthetic generation: inspect `outdir`; never trust a default or
  unreviewed path.
- After safe experiments: audit `git status --short` for generated files and
  remove only artifacts created by that experiment.
- Changes to producers require review through [change impact](change-impact.md),
  [source map](source-map.md), [build and install](build-and-install.md), and
  [synthetic comparison](testing/synthetic-comparison.md).

Evidence limits: no build, install, header regeneration, notebook execution, or
synthetic runtime was performed for this page.
