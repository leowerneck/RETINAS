# Documentation And Experiments

## Purpose

Canonical owner for README and notebook authority, reproducibility boundaries,
and documentation drift. Documentation expresses intended use or derivation;
it does not override current implementation, manifests, tests, or fresh
execution.

## Ground Truth

| Evidence | Role |
| --- | --- |
| `README.md` | Installation examples, dependency claims, synthetic-test instructions, output descriptions, and offline-usage intent |
| `doc/Interfacing_with_the_C_library.ipynb` | Stored Python experiment concerning generated images and displacement plotting |
| `doc/Upsampling.ipynb` | Stored derivation and Python experiments for one-dimensional/global and localized upsampling |
| `retinas/utils.py` | Current utility signatures and destructive synthetic generator |
| `retinas/pyretinas.py`, `retinas/retinas.py` | Current Python and bridge behavior to compare with prose/examples |
| Meson manifests and native sources | Current build selection and native definitions |
| Git history | Historical context for why prose/cells changed; never current runtime proof |

Jupyter's official [notebook-format description](https://nbformat.readthedocs.io/en/latest/format_description.html)
defines a notebook as metadata plus ordered cells. Code cells can retain an
`execution_count` and stored `outputs`; those serialized fields report notebook
state, not a fresh RETINAS execution in the present environment.

## Current Contract

### Authority Classes

| Material | What it can support | What it cannot support alone |
| --- | --- | --- |
| README prose and commands | Documented dependencies, intended entry points, expected use and outputs | Successful configuration, build, install, load, execution, or parity |
| Notebook Markdown/equations | Intended explanation or derivation | Match to current algorithm or numerical correctness |
| Notebook code cells | Experiment inputs, imports, transformations, and output paths | Clean-kernel reproducibility or current API compatibility |
| Stored execution counts | A serialized record that cells had counts when saved | Present execution order, success, environment, or freshness |
| Stored outputs | Previously serialized display/stream/result data | Current result, assertion, benchmark, or parity proof |
| Fresh named run | Runtime evidence for its exact environment and inputs | General support outside that scope |

### README

`README.md` is the prose entry point. Its configure/Make examples and dependency
lists document intended use; [build and install](build-and-install.md) owns
manifest-selected behavior and proof stages. Its synthetic instructions select
C, CUDA, and Python together and describe generated files; the
[synthetic comparison](testing/synthetic-comparison.md) owns prerequisites,
side effects, and oracle limits.

The offline section says the algorithm is only available in Python. Current C
and CUDA manifests select offline accumulation/update definitions, and the
bridge binds corresponding symbols. This is a documentation-versus-surface
discrepancy, not proof of supported native behavior; intended availability is
unresolved. The example's reset comments also differ from current pure-Python
code, which resets the sum to `None` and the counter to zero.

### Notebook Inputs, Cells, And Outputs

`doc/Interfacing_with_the_C_library.ipynb` declares a Python 3 kernel and
contains code cells but no Markdown narrative or stored outputs. Four cells
have non-null stored execution counts; one is unexecuted. Visible inputs include
NumPy, Matplotlib, `utils.Poisson_image`, plotting constants, and a relative
`../src/displacements.txt` load. Cells write `test.pdf`.

The current notebook passes `background`, `radius`, and `dotradius` keyword
arguments that current `Poisson_image` does not accept, and its referenced
tracked `src/displacements.txt` path is absent. Its filename, cell counts, and
historical execution counts do not establish current C-library interfacing or
reproducibility.

`doc/Upsampling.ipynb` declares a Python 3 kernel. Markdown cells explain
zero-padding and a localized DFT construction; code cells provide experiments
using NumPy, Matplotlib, and IPython display. Seven code cells have non-null
stored execution counts and four stored outputs. Multiple cells overwrite
`test.png`. Visible cells call names including `fftfreq` and `roll` without a
visible import in the notebook, so clean-kernel reproducibility is unresolved.

For either notebook, classify each claim separately:

1. inputs: parameters, imports, relative data paths, and current external files;
2. executable cells: source that could be run after prerequisites and paths are
   established;
3. stored outputs/counts: serialized historical notebook state;
4. derived explanation: Markdown, equations, and interpretation; and
5. runtime proof: only a fresh, named, observed execution with an oracle.

Do not execute or rewrite notebooks during KB maintenance. Do not copy their
cells or stored outputs into wiki prose. Any future run needs a disposable
working directory, pinned environment, inspected writes, clean-kernel order,
and an explicit success oracle.

## Verification And Impact

Verify documentation statically by comparing each material statement with the
claim-specific implementation, declaration, manifest, caller, or test. Inspect
notebook JSON for cell type, source, execution count, outputs, metadata, and
paths without executing it. Use history only to investigate provenance after
current evidence is understood.

Review this page after changes to `README.md`, `doc/*.ipynb`, mentioned Python
APIs, build/install routing, synthetic generation, or algorithm contracts. Also
review [generated boundaries](generated-boundaries.md),
[registration pipeline](algorithms/registration-pipeline.md), and
[coverage and gaps](testing/coverage-and-gaps.md). Confirmed prose/source
conflicts are bounded in [contradictions](contradictions.md).
