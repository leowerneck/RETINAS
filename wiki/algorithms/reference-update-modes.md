# Reference Update Modes

## Purpose

Own first-reference, online/IIR, accumulated/offline, averaged replacement,
reset, and continuation contracts. Backend allocation detail stays with
implementation owners; this page records only cross-mode state transitions.

## Ground Truth

| Layer | Exact paths and symbols |
| --- | --- |
| Pure Python | `retinas/pyretinas.py`: both first-reference methods, both update methods, both add methods, both averaged-replacement methods, both composites |
| Alignment helper | `retinas/utils.py`: `freq_shift` |
| Bridge | `retinas/retinas.py`: both composite methods, `update_reference_image_from_image_sum` |
| C transition evidence | `state_initialize.c`, `set_first_reference_image.c`, `initialize_image_sum.c`, `add_new_image_to_sum.c`, `update_reference_image.c`, `update_reference_image_from_image_sum.c`, both composite units, C `retinas.h` and `meson.build` |
| CUDA transition evidence | `state_initialize.cu`, `set_first_reference_image.cu`, `add_new_image_to_sum.cu`, `update_reference_image.cu`, `update_reference_image_from_image_sum.cu`, normal/shot-noise composite units |
| Intended-use prose/caller | Offline section in `README.md`; online path in `retinas/synthetic_data_test.py` |

All named native files above are in their respective C or CUDA backend source
directories linked through [source map](../source-map.md).

## Current Contract

### Shared Mode Vocabulary

| Transition | Observed operation |
| --- | --- |
| First reference | First preprocessed frame becomes normal-frequency or reciprocal-frequency reference; composite returns centering offsets without correlation |
| Online/IIR | Estimate displacement, align current representation, replace reference with `A0 * aligned_current + B1 * old_reference` |
| Accumulated/offline | Estimate against current reference, then add aligned current representation to image sum and increment counter; reference stability differs by implementation |
| Averaged replacement | Divide sum by counter into reference, then reset sum/counter according to implementation |

`Pyretinas` owns its counter and arrays as Python attributes. C/CUDA own them
inside native state; bridge owns only `first_image` and invokes native
transitions. Mixing online and offline composites on one state has no documented
contract and remains unresolved.

### Pure-Python First Frame And Online Mode

First normal reference is direct assignment from `new_image_freq`; shot-noise
reference is direct assignment from `reciprocal_new_image_freq`. The online
composite sets `first_image=False` and returns centering offsets. On later calls,
`freq_shift` multiplies current frequency data by positive-exponent horizontal
and vertical phase factors, then update replaces reference using `A0`, `B1`.

`time_constant` construction defines `A0=1-exp(-1/time_constant)` and
`B1=exp(-1/time_constant)`. With `time_constant=None`, pure Python does not
create these attributes; first frame still succeeds, but a later online update
reaches missing coefficients. Desired validation or no-update semantics are
unresolved.

### Pure-Python Offline Aliasing And Reset

Constructor starts `image_counter=1`, but first offline composite explicitly
reassigns `image_sum_freq` to the reference ndarray. No copy is made, so
reference and sum share identity in both modes. Second-frame `+= aligned_current`
mutates that reference in place as well as the sum; counter becomes two. This
means later frames no longer compare against an unchanged first-frame reference,
despite README fixed-reference prose. Intent is unresolved.

Averaged replacement creates a new reference from `image_sum_freq/image_counter`,
then sets `image_sum_freq=None`, `image_counter=0`; it does not set
`first_image=True`. The next offline composite follows the non-first path and
eventually evaluates `None += aligned_current`, producing `TypeError`. Both
normal and shot-noise paths have this transition. Current docstrings and README
instead describe resetting sum to reference and counter to one; implementation
behavior is authoritative, intended continuation remains unresolved.

### Native Cross-Reference

C state allocation reserves `image_sum_freq` but does not initialize its
contents or `image_counter`. C first-reference setup initializes only reference.
`initialize_image_sum.c` would copy reference and set counter one, and is
build-selected, but no declaration or caller is found in the C header, C units,
or bridge. Bridge first offline call only invokes first-reference setup and
returns; second call reaches C `add_new_image_to_sum`, which adds to and
increments this unresolved state. Deep allocation/API analysis belongs to the C
implementation owner; no safety, defect-intent, or runtime conclusion follows
from this static chain.

CUDA initializes counter to zero at state creation. First-reference setup then
enqueues a reference-to-sum copy and sets the host counter to one; the path has
no synchronization or launch-status check proving copy completion before host
return. Averaged replacement divides sum into reference, zeroes sum, and resets
counter zero; a later addition uses the zeroed sum and increments counter. This
is defined/build-selected evidence, not execution or parity proof.

C averaged replacement copies average into both reference and sum and resets
counter one. These C/CUDA reset differences and Python `None` reset preclude a
shared continuation claim.

## Verification And Impact

Static verification traced first and second offline calls, object assignment
versus in-place mutation, counter changes, averaged replacement, and next-cycle
control flow. A no-bytecode in-memory probe for both Python modes observed alias
identity and post-reset `TypeError`; it is `executed` in that local environment
but not `assertion-covered`. C/CUDA findings are static
`defined`/`build-selected`; neither backend was built or run.

`README.md` is `documented` offline intent. `retinas/synthetic_data_test.py` is
`selected-by-test/example` for online normal mode only and has no state or
tolerance assertions. Needed tests: object identity before/after first `+=`,
fixed-reference oracle across at least three frames, post-reset next cycle,
`time_constant=None`, and C first/second offline calls in a named precision and
environment.

Review this page for reference, alignment, sum, counter, first-frame, reset, or
composite changes. Also review [pure Python](../implementations/python.md),
[registration pipeline](registration-pipeline.md),
[displacement estimation](displacement-estimation.md),
[testing](../testing/index.md), and [change impact](../change-impact.md).
