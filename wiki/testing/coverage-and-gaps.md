# Coverage And Gaps

## Purpose

Canonical owner for current verification strength and high-value missing proof.
This page records bounded gaps and smallest useful experiments; it does not
invent a supported product matrix or treat shared names as parity.

## Ground Truth

| Evidence area | Exact paths |
| --- | --- |
| Test/example selection | `retinas/synthetic_data_test.py`, `README.md` |
| Pure-Python lifecycle | `retinas/pyretinas.py` |
| Bridge selection/lifecycle | `retinas/retinas.py` |
| C offline state | C `state_initialize.c`, `set_first_reference_image.c`, `initialize_image_sum.c`, `add_new_image_to_sum.c`, composite unit, update-from-sum unit, header, manifest |
| CUDA offline state | Corresponding CUDA units, `retinas.h`, `function_prototypes.h`, manifest |
| Registration and automation | `configure`, `meson.build.in`, all recursive Meson manifests, tracked repository paths |
| Cross-layer flow | [Registration pipeline](../algorithms/registration-pipeline.md) |

Test evidence requires an assertion/oracle, registration or named runner, and
observed execution in the stated environment. Source presence, compilation,
result files, and historical execution claims are weaker and stay distinct.

## Current Contract

### Observed Coverage Boundary

`retinas/synthetic_data_test.py` is selected-by-example for C, CUDA, and Python
normal online-update paths with native single precision. It uses random data,
writes numerical results, and has no assertions or tolerance. It does not
select shot noise, offline accumulation/update cycles, double precision, or
backend-isolated operation. No tracked Meson `test()` registration,
pytest/unittest product suite, or product-test CI job was found. Tracked
`.github/workflows/kb.yml` validates KB structure/tooling, not RETINAS product
numerics.

Keep these stages separate for every claim:

1. **configuration**: named options and dependencies completed setup;
2. **build**: named source/target compiled and linked;
3. **load**: named artifact opened and requested symbols bound;
4. **execution**: named callable completed with stated fixture;
5. **result**: output shape/value/file was observed;
6. **assertion**: an explicit oracle and tolerance passed; and
7. **CI**: a tracked automation route selected that assertion in a named job.

Later stages do not follow from earlier ones. CI presence would not by itself
prove a job ran; a result file would not by itself prove correctness.

### High-Value Gaps

#### C First And Second Offline Accumulation

C `state_initialize` allocates `image_sum_freq` without visibly initializing it
or `image_counter`. `set_first_reference_image` initializes only reference
frequency data. Manifest-selected `initialize_image_sum.c` would copy reference
into the sum and set counter one, but no C-header declaration, bridge binding,
or call site was found. Bridge first offline call sets the reference and
returns; its next call reaches C accumulation.

Smallest proof: in a disposable C/bridge test, process first then second known
frames through offline mode and assert initialized sum/counter state plus a
finite, expected displacement/accumulation result. Run under a memory checker
when available. Until then, first-to-second C offline behavior is unresolved,
not parity-confirmed.

#### Pure-Python Continuation After Reset

Pure-Python averaged-reference update assigns a new reference, then sets
`image_sum_freq = None` and counter zero while `first_image` remains false. The
next offline composite follows the non-first path and its add method performs
in-place addition into `image_sum_freq`.

Smallest decision and proof: maintainer defines whether a new accumulation
cycle should seed from the updated reference or start empty; a normal-mode and
shot-noise test then completes update followed by one more frame and asserts
counter, accumulator, reference, and displacement behavior. Current
post-reset continuation is unresolved.

#### Reference/Sum Alias Mutation

On first pure-Python offline frame, `image_sum_freq` is assigned directly from
`ref_image_freq` or `reciprocal_ref_image_freq` without a visible copy. Later
`+=` can therefore mutate storage shared with the reference before explicit
reference replacement.

Smallest proof: assert object-storage identity immediately after first-frame
setup, retain an independent expected reference, add a known second frame, and
assert whether reference mutation matches an explicitly approved contract.
Cover normal and shot-noise branches separately; do not infer independence from
variable names.

### Backend, Precision, Mode, And Tolerance Unknowns

No current assertion evidence establishes C/CUDA/Python equivalence for any
fixture. CUDA runtime proof may be unavailable on a given host. C shot-noise
symbols requested by the bridge are not found in current C declarations,
definitions, or source list. Single versus double precision, online versus
offline update, normal versus shot noise, first-image centering choice, and
upsampling factor can change expected numeric behavior.

Any future parity claim must name implementations, exact fixture/seed, backend,
precision, mode, centering, update lifecycle, expected result, absolute and/or
relative tolerance, environment, and observed assertion run. Select tolerances
from an approved numerical contract and measured error behavior, not from a
guessed universal matrix.

### Automation Gap

Current Makefile generation exposes `check` commands that call `meson test`,
but tracked manifests register no product tests. Structural KB unittest coverage
under `scripts/tests/` and tracked KB CI are separate and say nothing about
RETINAS numerics.
Choosing product CI policy, supported combinations, fixture budgets, GPU
availability policy, and required tolerances remains a maintainer decision.

## Verification And Impact

For queries, report only the highest observed evidence stage and its exact
backend/mode/precision/environment. For changes, trace caller, declaration,
definition, manifest, runner, fixture, oracle, tolerance, and observed result;
record missing layers as `absent`, `not found`, or `unresolved`.

Review this page after changes to accumulation/reset state, bridge dispatch,
backend source/header lists, synthetic driver, test registration, CI paths, or
approved numeric contracts. Also review [synthetic comparison](synthetic-comparison.md),
[testing hub](index.md), and [registration pipeline](../algorithms/registration-pipeline.md).
