# RETINAS Testing And Evidence

## Purpose

Define verification vocabulary, proof boundaries, and safe routing. This hub
does not turn test-like source or successful execution into parity proof.

## Read First

| Evidence need | Route | Primary evidence |
| --- | --- | --- |
| Knowledge-base structure | [KB checks](../lint/CHECKS.md) | `scripts/kb.py`, `scripts/tests/test_kb.py` |
| Manual three-way comparison | [Synthetic comparison](synthetic-comparison.md) | `retinas/synthetic_data_test.py`, `retinas/utils.py`, `README.md` |
| Current coverage and smallest gaps | [Coverage and gaps](coverage-and-gaps.md) | callers, runners, oracles, observed execution |
| Cross-backend evidence | [Parity matrix](../implementations/parity-matrix.md) | implementation owners and exact tests |
| Algorithm/state path | [Registration pipeline](../algorithms/registration-pipeline.md) | exact implementation and caller |
| Build/runtime claim | [Build owner](../build-and-install.md) plus named backend owner/environment | manifests, command, output, oracle |

Proof levels remain separate: documented intent; test/example source selects a
path; runner registration; configured/compiled/linked; loaded; executed; and
assertion-covered. Numerical parity additionally requires named implementations,
mode, precision, fixture, tolerance/oracle, and execution environment.

## Common Tasks

- Before running synthetic generation, read
  [generated boundaries](../generated-boundaries.md); its output directory is
  removed and recreated.
- For a changed contract, identify smallest deterministic unit assertion first,
  then integration/runtime checks only as needed.
- For unavailable CUDA/native dependencies, report static evidence and explicit
  execution gap; never downgrade absence of environment into product failure.
- For KB changes, run validator tests and lifecycle-appropriate check.

## Ownership Boundary

This hub owns vocabulary and routing. Synthetic and coverage leaves own fixture,
output, and gap semantics. Product implementation truth stays in source;
algorithm flow stays under algorithms; backend claims stay under
implementations.

## Evidence Limits And Current Gaps

No Meson `test()` registration or conventional pytest/unittest product suite is
visible. Tracked `.github/workflows/kb.yml` validates KB structure/tooling; no
tracked CI job selects product tests. `retinas/synthetic_data_test.py` generates
1,000 post-reference moves plus the initial frame, 1,001 images total, loads
fixed local library paths, writes diagnostics/results, and compares printed
values without an assertion oracle. Static selection is not execution or
parity confirmation.

Parent: [wiki index](../index.md). Related: [source map](../source-map.md),
[change impact](../change-impact.md), and [implementations](../implementations/index.md).
