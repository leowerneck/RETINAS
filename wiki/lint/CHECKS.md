# Knowledge-Base Checks

## Purpose

Define dependency-free integrity checks for RETINAS knowledge-base Markdown.
These checks validate structure and routing, not product semantics or numerical
correctness.

## Manual Bootstrap Checks

From repository root:

```sh
if rg -n '[[:blank:]]+$' AGENTS.md wiki scripts; then exit 1; fi
rg -n '(^|[^[:alnum:]_.])/work(/|`|$)|file://' AGENTS.md wiki || true
rg -n '\]\((/|file:|\.\./\.\.)' AGENTS.md wiki || true
rg -n 'GRHayL/|plan[^ ]*\.md|tasks[0-9]*\.md' AGENTS.md wiki
git ls-files | sort
```

Review every hit: exclusion/policy prose is allowed; actual links and authority,
inventory, or impact targets into excluded material are forbidden. Follow every
root/index/router link. Confirm every live wiki page appears exactly once in
`wiki/index.md`, except index itself. Reconcile source-map families against
`git ls-files`. Ensure every file ends with one newline.

## Automated Check

Staged bootstrap:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 scripts/kb.py check --bootstrap
test -z "$(find scripts \( -type d -name __pycache__ -o -type f -name '*.py[co]' \) -print)"
```

Completed ownership uses `python3 scripts/kb.py check` without `--bootstrap`.
The checker discovers only root `AGENTS.md` and live `wiki/**/*.md`. It never
rewrites content or requires product/GPU/native dependencies.

Authority existence comes from current worktree files that are also present in
Git's index, including staged additions. Untracked product, build, docs,
exemplar, plan, output, and cache files cannot satisfy `Ground Truth`, source,
or impact authority merely by existing. One local-bootstrap exception applies
before `AGENTS.md` first enters the index: only live `AGENTS.md`, discovered
`wiki/**/*.md`, `scripts/kb.py`, `scripts/tests/test_kb.py`, and
`.github/workflows/kb.yml` may satisfy their own KB/tooling authority rows.
Indexing `AGENTS.md` ends that exception. Temporary test roots stage their
fixture graph before checking it.

Advisory impact routing accepts explicit paths or Git endpoints:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 scripts/kb.py impact README.md retinas/retinas.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/kb.py impact --base HEAD^
PYTHONDONTWRITEBYTECODE=1 python3 scripts/kb.py impact --base HEAD^ --head HEAD
```

The command parses only the strict table in `wiki/change-impact.md`. It reports
all overlapping routes, exclusions, unmapped paths, owners, and verification;
it edits nothing and makes no semantic currency claim.

## Advisory CI Contract

`.github/workflows/kb.yml` runs on every pull request, every push, and manual
dispatch. Full checkout history supports explicit endpoint
resolution. Pull requests use payload base/head commits; ordinary pushes use
available nonzero before/after commits. New branches and unavailable before
commits use the default-branch merge base, or Git's empty tree when no common
ancestor exists. Deleted refs skip tree checks. Manual runs use a supplied base
or a validated `HEAD^`; a root commit without input fails with guidance.

Only a KB/tooling change runs blocking default `kb.py check` and scoped
`git diff --check`. Impact output remains advisory and runs for mapped product,
build, docs, and KB changes. This workflow does not run product builds or make
itself a required check. Ground truth: Python's
[`unittest` documentation](https://docs.python.org/3/library/unittest.html),
Git's [`git diff` documentation](https://git-scm.com/docs/git-diff), GitHub's
[workflow syntax](https://docs.github.com/actions/writing-workflows/workflow-syntax-for-github-actions),
[context reference](https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs),
[webhook payload reference](https://docs.github.com/en/webhooks/webhook-events-and-payloads),
and official [`actions/checkout`](https://github.com/actions/checkout).

## Enforced Contracts

- paths stay inside repository; inline Markdown links, reference definitions,
  and raw HTML `href`/`src` targets resolve relative to source;
- no actual authority/link target enters excluded exemplar or plan/task files;
- root reaches every live page; index coverage is exhaustive and kinds are
  controlled;
- page titles and catalog aliases are unique;
- indexed page kinds provide required headings;
- designated authority paths exist in both worktree and Git index; declarations
  under `Expected Generated Outputs` do not establish authority;
- source/impact/log tables follow documented shapes;
- staged and complete ownership lifecycle rules hold;
- whitespace, final newline, fingerprint assignments, and stored freshness
  metadata are rejected.

Policy prose may say: Do not store hashes or mtimes. Unsupported Markdown forms
remain manual-review territory and do not create graph reachability. Recognized
reference definitions and raw HTML link attributes are never silently skipped.

## Semantic Review

After structural success, reopen changed/high-blast-radius sources. Challenge
support/parity claims, duplicated explanations, missing callers/tests, unsafe
generated commands, and stale conflict wording. Structural success does not
prove runtime behavior, build success, or numerical parity.
