# RETINAS Agent Knowledge Base

## Purpose And Scope

This file is RETINAS's root instruction schema and task router. Persistent
synthesis lives in [`wiki/`](wiki/index.md); tracked source, manifests, tests,
and docs remain evidence. Reopen exact evidence before material claims or
changes.

Scope covers tracked RETINAS product, build, documentation, and knowledge-base
files. It excludes the `GRHayL/` exemplar, planning/task files, `.git/`, build
and install trees, caches, notebook outputs, synthetic images, diagnostics, and
results. Do not use excluded material as RETINAS authority or inventory.

Root instructions apply to this repository tree. Codex reads instruction files
from project root toward the working directory; a deeper applicable file is
more specific and takes precedence on conflict. Direct system, developer, and
user instructions remain higher priority. This v1 creates no nested
`AGENTS.md`; add one only when a subtree needs materially narrower automatic
instructions. Sources: [official Codex guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md.md)
and the [AGENTS.md interoperability specification](https://agents.md/).

## Authority And Evidence

Match authority to claim:

- Runtime behavior/state: exact `.py`, `.c`, or `.cu` implementation and direct
  callers; executed tests strengthen the claim for their named environment.
- Native declaration: relevant header; definitions, installed output, and
  bindings corroborate different layers.
- Python API: `retinas/pyretinas.py`, `retinas/retinas.py`, and
  `retinas/utils.py`; README/examples express usage or intent.
- Build inclusion: `meson.build.in`, recursive `meson.build` files,
  `meson_options.txt`, and `configure` for the named entry route.
- Install surface: install declarations plus an observed disposable-prefix
  install.
- Test coverage: assertions and registration/runner plus observed execution.
- Intended usage/derivation: README, notebooks, public docs, and comments,
  checked against implementation and tests.
- Fresh execution: exact command, environment/mode, exit/result, and oracle.

Wiki prose is derived navigation and synthesis. On conflict, trust current
claim-specific repository evidence, report the discrepancy, and update wiki
only within authorization. External sources support upstream/tool/algorithm
facts; they never override observed RETINAS behavior.

Availability and parity statements use: `documented`, `declared`, `defined`,
`build-selected`, `bound`, `compiled/linked`, `selected-by-test/example`,
`executed`, `assertion-covered`, `inferred`, or `unresolved`. Use `absent`,
`not found`, `not applicable`, or `unknown` instead of blank support cells.
`Parity-confirmed` requires named implementations, mode, precision, fixture,
tolerance/oracle, and execution evidence.

Safe unknown wording: state observed layer, name missing evidence, use
`unresolved`, and request the smallest coherent maintainer decision or
experiment. Do not infer support from presence, build inclusion from a
declaration, execution from compilation, or parity from shared names.

## Start Here

- [Wiki index](wiki/index.md): exhaustive live-page inventory and task starts.
- [Catalog](wiki/catalog.md): term, symbol, alias, and question routing.
- [Source map](wiki/source-map.md): tracked path families and primary owners.
- [Generated boundaries](wiki/generated-boundaries.md): generated, runtime,
  destructive, and unknown-provenance boundaries.
- [Architecture](wiki/architecture.md): thin layer/component router.
- [Change impact](wiki/change-impact.md): changed paths to review routes.
- [Algorithms](wiki/algorithms/index.md): shared algorithm vocabulary.
- [Registration pipeline](wiki/algorithms/registration-pipeline.md): canonical
  end-to-end frame and state flow.
- [Implementations](wiki/implementations/index.md): backend vocabulary and
  ownership boundaries.
- [Public API map](wiki/public-api-map.md): layered callable/install evidence.
- [Parity matrix](wiki/implementations/parity-matrix.md): cross-backend evidence
  without inferred equivalence.
- [Build and install](wiki/build-and-install.md): configuration, targets,
  dependencies, installation, and cleanup.
- [Docs and experiments](wiki/docs-and-experiments.md): README/notebook evidence
  and reproducibility boundaries.
- [Testing](wiki/testing/index.md): evidence-strength and verification routing.
- [Contradictions](wiki/contradictions.md): confirmed competing evidence only.
- [Workflows](wiki/workflows.md): product-change and KB operation playbooks.
- [Operation log](wiki/log.md): material KB operations, not technical facts.
- [KB checks](wiki/lint/CHECKS.md): manual and automated integrity checks.

## Task Router

| Task | Start | Reopen next |
| --- | --- | --- |
| Locate source ownership | [Source map](wiki/source-map.md) | Exact listed files and manifests |
| Understand system layers | [Architecture](wiki/architecture.md) | Component owner and exact evidence |
| Trace registration lifecycle | [Registration pipeline](wiki/algorithms/registration-pipeline.md) | Python/bridge/native implementation named there |
| Change shared algorithm | [Algorithms](wiki/algorithms/index.md) | Pipeline, backend owner, tests, impact routes |
| Change Python/C/CUDA | [Implementations](wiki/implementations/index.md) | Source map, relevant algorithm, tests |
| Investigate parity/lifecycle | [Registration pipeline](wiki/algorithms/registration-pipeline.md) | Evidence layer for each implementation; do not assume parity |
| Add or inspect public symbol | [Public API map](wiki/public-api-map.md) | Declaration, definition, manifest, binding, install, caller, test |
| Change build/install/docs | [Workflows](wiki/workflows.md) | Build/docs owner, generated boundary, exact evidence |
| Resolve conflicting evidence | [Contradictions](wiki/contradictions.md) | Both owners and smallest decision/experiment |
| Inspect generated/output safety | [Generated boundaries](wiki/generated-boundaries.md) | Generator or destructive caller before running |
| Run or assess tests | [Testing](wiki/testing/index.md) | Exact runner, fixture, oracle, environment |
| Ingest/query/file/lint/curate | Operations below | Index, catalog, impact, log as applicable |

For any unlisted term or changed family, start at catalog/source map and follow
current evidence. Do not create a placeholder route.

## Page Types And Placement

- `hub`: purpose, child/read-first routes, common tasks, boundary, gaps, parent
  and cross-links. Navigation summary only.
- `leaf`: canonical owner. Required headings are `Purpose`, `Ground Truth`,
  `Current Contract`, and `Verification And Impact`.
- `map`: declares strict table schema near top; rows link one canonical owner
  and exact evidence. Detailed behavior stays in leaves. Contradiction rows add
  competing evidence, safest wording, impact, owner, and smallest experiment.
- `special`: index, catalog, log, or checks with its documented contract.

Edit an existing canonical owner first. Create a page only when the concept has
an independent contract, helps a benchmark question, and will have at least two
meaningful inbound routes. Land page, index entry, parent route, catalog alias
when needed, source ownership, and impact route atomically. Never create one
page per file/function or symmetry-only placeholders.

Shared algorithm contracts belong under `wiki/algorithms/`; backend state under
`wiki/implementations/`; comparisons in the canonical parity map; public
surface layering in the API map; bridge ABI/lifetime in its bridge
owner; build/output facts in build/generated owners; test intent/oracles in
testing; unresolved conflicts in one contradiction registry.

Bootstrap budget: root 150-230 lines, hubs normally at most 120, leaves normally
at most 220, no more than 30 content pages, and at most 4,500 Markdown lines
across root plus wiki. Stop and request curation when a limit or new-page rule
fails.

## Operations

### Ingest

1. Confirm authorized scope and changed paths; preserve unrelated changes and
   exclude exemplars, plans, generated output, and caches.
2. List five recent log headings with
   `rg '^## ' wiki/log.md | tail -n 5`; open only relevant entry blocks.
3. Read root, index/catalog, impact route, owner, exact source, declaration,
   manifest/config, caller, docs, and closest test as claim requires.
4. Classify evidence; update canonical owner first; record conflicts without
   inventing intent.
5. Run structural and proportionate safe checks, audit whole diff, then append
   one log entry only for successful material mutation or resolved decision.

### Query

Start at catalog for a term/question or index for an area; use `rg` if absent.
Read smallest owner set, reopen source/build/test for material claims, and answer
with evidence level, backend/mode/precision, and limits. Query is read-only by
default. Ignore log unless question concerns recent KB work.

### Filed Query

When KB maintenance is authorized and query yields durable multi-source
synthesis, update an existing owner. A new page must satisfy placement and land
all inbound routes. Separate evidence from inference, run ingest checks, and log
only a successful material mutation.

### Lint

Run deterministic checks first. Then review changed/high-blast-radius evidence,
availability/parity wording, duplicated explanation, missing callers/tests,
unsafe commands, and stale contradiction wording. Structural errors block;
semantic uncertainty becomes a review item until precisely testable.

### Change Sync

Use [change impact](wiki/change-impact.md), reopen exact changed evidence, and
record `updated`, `still accurate`, or `not applicable` with reason outside KB.
Edit KB and log only when routing or durable synthesis changed.

### Curate

Human owns ontology, supported-product intent, contradiction resolution,
external-source selection, page split/merge, and CI rule promotion. Agent may
perform authorized source-backed edits, link repair, lint, and one concise log
entry after a decision. This follows the persistent, interlinked, compounding
Markdown pattern described in Karpathy's primary
[LLM Wiki idea file](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Write And Safety Policy

- Link and synthesize; do not copy source listings, long prototypes/comments,
  README passages, notebook cells, raw output, or diffs.
- Do not modify product code, build behavior, docs, or public APIs during KB
  maintenance without separate explicit authorization.
- Never edit generated root manifests, Makefiles, build/install products,
  notebook outputs, caches, synthetic images, diagnostics, or result files as
  KB content.
- Inspect any destructive output path before execution. Synthetic generation
  removes and recreates its chosen output directory.
- Treat generated-warning native headers as unknown provenance until a current
  repository command is proven; do not invent or run a generator.
- Do not require CUDA, FFTW, BLAS, SciPy, Meson, or product build merely to lint
  Markdown. Use disposable environments for build/install/runtime claims.
- Do not store hashes or mtimes. No source counts as freshness, verification
  badges, copied command dumps, or absolute workspace/file URLs.

## Completion Checks

During staged expansion run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 scripts/kb.py check --bootstrap
test -z "$(find scripts \( -type d -name __pycache__ -o -type f -name '*.py[co]' \) -print)"
```

After ownership becomes complete, omit `--bootstrap`. Also run proportionate
product checks only for claims/change scope and in a safe environment. Report
unavailable native/CUDA proof as a gap. Human approval remains required for
ontology, product intent, contradiction resolution, CI policy, merge, or deploy.
