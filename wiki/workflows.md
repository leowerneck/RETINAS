# RETINAS Change And Knowledge Workflows

## Purpose

Route common product and KB changes through canonical owners, evidence, safety
checks, and impact review. Root `AGENTS.md` remains authoritative for ingest,
query, filed-query, lint, change-sync, and curation policy.

## Ground Truth

| Workflow input | Exact evidence and route |
| --- | --- |
| Changed tracked family | `wiki/change-impact.md`, `wiki/source-map.md` |
| KB operations and log policy | `AGENTS.md`, `wiki/log.md` |
| Product verification | `wiki/testing/index.md`, `wiki/testing/coverage-and-gaps.md` |
| Generated/destructive work | `wiki/generated-boundaries.md` |
| Build/install work | `configure`, `meson.build.in`, `meson_options.txt` |

## Current Contract

### Product Change Playbooks

| Change | Start and reopen | Minimum review and verification |
| --- | --- | --- |
| Pure Python or helper | [Pure Python](implementations/python.md), then relevant algorithm leaf and exact caller | Parse/import safely; focused deterministic state/numeric test; review parity and API map |
| Bridge ABI or lifecycle | [Ctypes bridge](interfaces/ctypes-bridge.md), both backend declarations/definitions/manifests | Match precision and every `argtypes`/`restype`; missing-symbol and finalization tests in disposable native build |
| C backend | [C implementation](implementations/c.md), header, manifest, bridge, closest caller | Static reconciliation first; named precision build/runtime only in disposable environment; explicit oracle |
| CUDA backend | [CUDA implementation](implementations/cuda.md), headers, manifest, bridge | Static reconciliation first; named GPU/toolchain/mode/precision for execution; inspect API errors and oracle |
| Native utilities | [Native utilities](implementations/native-utilities.md), local Makefile, main Meson manifests | Reconcile header/caller/build wiring before compile or availability claim |
| Shared algorithm | Relevant [algorithm](algorithms/index.md), then pipeline, backend owners, parity, tests | Preserve one stage owner and one end-to-end owner; fixture names sign, precision, mode, tolerance |
| Public symbol | [Public API map](public-api-map.md), declaration, definition, manifest, binding/install/caller | Verify every evidence layer separately; do not infer support from declaration |
| Build/install | [Build and install](build-and-install.md), exact configuration and recursive manifests | Use disposable build/prefix; record backend/precision/toolchain and inspected artifacts |
| Test or synthetic driver | [Testing](testing/index.md), fixture, runner, oracle, output paths | Inspect deletion target; distinguish selection, execution, result, assertion, and CI |
| README or notebook | [Docs and experiments](docs-and-experiments.md), then claim-specific implementation/build/test | Static notebook JSON review by default; no stored output as fresh proof; do not execute/rewrite outputs |
| Generated/runtime output | [Generated boundaries](generated-boundaries.md), producer and destructive caller | Resolve exact path before execution; use disposable location; audit only created artifacts |
| Python lint policy | `retinas/.pylintrc`, affected Python files, established lint invocation | Treat config as workflow policy; do not claim a passing lint run without named command/result |

### Knowledge-Base Playbooks

- **Ingest:** confirm scope and changed paths; read five recent log headings and
  only relevant blocks; follow source map/impact route; reopen exact evidence;
  update canonical owner first; validate/diff-audit; log one successful material
  mutation.
- **Query:** start at catalog/index, use `rg` when absent, reopen exact evidence,
  answer with evidence stage and limits; remain read-only unless maintenance is
  authorized.
- **Filed query:** write durable multi-source synthesis to existing owner first;
  new page requires independent contract, benchmark value, two inbound routes,
  and atomic graph/map updates.
- **Lint:** run deterministic checks, then inspect high-blast-radius evidence,
  availability/parity wording, duplicate narratives, callers/tests, destructive
  commands, and contradiction state.
- **Change sync:** use impact map; record `updated`, `still accurate`, or `not
  applicable` outside KB; edit/log only when durable synthesis changes.
- **Curate:** human decides ontology, product intent, contradictions, external
  sources, splits/merges, and CI promotion; agent may perform authorized
  source-backed repair and validation.

## Verification And Impact

Run lifecycle-appropriate KB checks after any KB mutation; once ownership is
complete, use strict default only. Product verification stays proportional and
safe. Every changed family must have exactly one primary owner plus applicable
`Also review` routes. Unknown paths trigger source-map reconciliation rather
than guessed ownership. Review this page after workflow, lint config, build,
generated-output, testing, or root operation-policy changes.
