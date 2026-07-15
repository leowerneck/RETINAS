"""Temporary-root contract tests for scripts/kb.py."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "kb.py"


class KBCheckTests(unittest.TestCase):
    maxDiff = None

    def make_root(self, *, staged: bool = True) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "wiki").mkdir()
        (root / "src").mkdir()
        (root / "src/app.py").write_text("VALUE = 1\n", encoding="utf-8")

        (root / "AGENTS.md").write_text(
            """# Fixture Root

## Purpose And Scope

Fixture policy may mention `GRHayL/`, `plan*.md`, and `tasks?.md` as exclusions.

## Authority And Evidence

Repository evidence wins.

## Start Here

- [Index](wiki/index.md)

## Task Router

Use index.

## Page Types And Placement

Use controlled kinds.

## Operations

Query, ingest, lint, and curate conservatively.

## Write And Safety Policy

Do not store hashes or mtimes.

## Completion Checks

Run checker.
""",
            encoding="utf-8",
        )

        (root / "wiki/index.md").write_text(
            """# Fixture Index

## Purpose

Exhaustive live page index.

## Pages

| Page | Kind | Purpose |
| --- | --- | --- |
| [Catalog](catalog.md) | special | Routes aliases. |
| [Impact](change-impact.md) | map | Routes changes. |
| [Log](log.md) | special | Records operations. |
| [Source map](source-map.md) | map | Owns source paths. |
| [Topic](topic.md) | leaf | Owns fixture behavior. |
""",
            encoding="utf-8",
        )
        (root / "wiki/catalog.md").write_text(
            """# Fixture Catalog

## Purpose

Route aliases.

## Routes

| Alias | Canonical owner | Evidence |
| --- | --- | --- |
| app behavior | [Topic](topic.md) | `src/app.py` |
""",
            encoding="utf-8",
        )
        owner = "`unassigned`" if staged else "[Topic](topic.md)"
        (root / "wiki/source-map.md").write_text(
            f"""# Fixture Source Map

KB ownership status: {'staged' if staged else 'complete'}

## Purpose And Schema

Strict inventory.

| Source family | Observed role | Primary owner | Exact evidence |
| --- | --- | --- | --- |
| `src/*.py` | Product source | {owner} | `src/app.py` |
| `AGENTS.md` | Root schema | [Topic](topic.md) | `AGENTS.md` |
| `wiki/**/*.md` | Wiki graph | [Topic](topic.md) | `wiki/index.md` |
""",
            encoding="utf-8",
        )
        impact_product = (
            "| `src/*.py` | [Topic](topic.md) | None | Unit test |\n"
            if not staged
            else ""
        )
        (root / "wiki/change-impact.md").write_text(
            """# Fixture Impact

## Purpose And Schema

Strict impact routing.

| Path glob | Primary owner | Also review | Verification |
| --- | --- | --- | --- |
"""
            + impact_product
            + """| `AGENTS.md` | [Topic](topic.md) | [Source map](source-map.md) | Run checker |
| `wiki/**/*.md` | [Topic](topic.md) | None | Run checker |
""",
            encoding="utf-8",
        )
        (root / "wiki/log.md").write_text(
            """# Fixture Log

## 2026-07-14 | bootstrap | fixture

- Scope: fixture
- Pages: all
- Result: graph created
- Verification: checker
- Follow-up: None
""",
            encoding="utf-8",
        )
        (root / "wiki/topic.md").write_text(
            """# Fixture Topic

## Purpose

Own fixture behavior.

## Ground Truth

| Evidence | Role |
| --- | --- |
| `src/app.py` | Product source |

## Current Contract

Value exists.

## Verification And Impact

Read source and run unit tests.
""",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "init", "-q"], cwd=root, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "add", "."], cwd=root, check=True, capture_output=True
        )
        return root

    def run_check(self, root: Path, *, bootstrap: bool) -> subprocess.CompletedProcess[str]:
        command = ["python3", str(SCRIPT), "check", "--root", str(root)]
        if bootstrap:
            command.append("--bootstrap")
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def run_impact(
        self,
        root: Path,
        *arguments: str,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(SCRIPT), "impact", "--root", str(root), *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )

    def commit(self, root: Path, message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-q",
                "-m",
                message,
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def run_ci_range(
        self,
        root: Path,
        event_name: str,
        payload: dict[str, object],
        sha: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
        event = root / "event.json"
        event.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "ci-range",
                "--root",
                str(root),
                "--event-name",
                event_name,
                "--event-file",
                str(event),
                "--sha",
                sha,
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_passes(self, root: Path, *, bootstrap: bool) -> None:
        result = self.run_check(root, bootstrap=bootstrap)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("KB check passed", result.stdout)

    def assert_fails(self, root: Path, fragment: str, *, bootstrap: bool) -> None:
        result = self.run_check(root, bootstrap=bootstrap)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(fragment, result.stdout + result.stderr)

    def test_staged_valid(self) -> None:
        self.assert_passes(self.make_root(staged=True), bootstrap=True)

    def test_staged_invalid_missing_inventory_coverage(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/source-map.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line for line in text.splitlines() if "`src/*.py`" not in line)
            + "\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "unmapped tracked path: src/app.py", bootstrap=True)

    def test_complete_valid(self) -> None:
        self.assert_passes(self.make_root(staged=False), bootstrap=False)

    def test_complete_invalid_missing_impact(self) -> None:
        root = self.make_root(staged=False)
        path = root / "wiki/change-impact.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line for line in text.splitlines() if "`src/*.py`" not in line)
            + "\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "missing impact route for source family: src/*.py", bootstrap=False)

    def test_marker_only_flip_does_not_complete_maps(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/source-map.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "KB ownership status: staged", "KB ownership status: complete"
            ),
            encoding="utf-8",
        )
        self.assert_fails(root, "complete source map contains unassigned owner", bootstrap=False)

    def test_policy_exclusion_mentions_and_negated_fingerprint_are_allowed(self) -> None:
        self.assert_passes(self.make_root(staged=True), bootstrap=True)

    def test_forbidden_exemplar_link_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Exemplar](../GRHayL/AGENTS.md)\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "forbidden link target", bootstrap=True)

    def test_reference_style_exemplar_link_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "\n[Exemplar][excluded]\n\n[excluded]: ../GRHayL/AGENTS.md\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "forbidden link target", bootstrap=True)

    def test_raw_html_exemplar_href_fails(self) -> None:
        for attribute in ("href", "src"):
            with self.subTest(attribute=attribute):
                root = self.make_root(staged=True)
                path = root / "wiki/topic.md"
                path.write_text(
                    path.read_text(encoding="utf-8")
                    + f'\n<a {attribute}="../GRHayL/AGENTS.md">excluded</a>\n',
                    encoding="utf-8",
                )
                self.assert_fails(root, "forbidden link target", bootstrap=True)

    def test_forbidden_plan_authority_target_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "| `src/app.py` | Product source |",
                "| `plan_synth.md` | Product source |",
            ),
            encoding="utf-8",
        )
        self.assert_fails(root, "forbidden authority target", bootstrap=True)

    def test_escaping_relative_link_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n[Escape](../../outside.md)\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "link escapes repository", bootstrap=True)

    def test_missing_ground_truth_evidence_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "`src/app.py`", "`src/missing.py`"
            ),
            encoding="utf-8",
        )
        self.assert_fails(root, "missing authority path: src/missing.py", bootstrap=True)

    def test_untracked_file_does_not_satisfy_ground_truth_authority(self) -> None:
        root = self.make_root(staged=False)
        (root / "scratch").mkdir()
        (root / "scratch/ghost.py").write_text("GHOST = True\n", encoding="utf-8")
        topic = root / "wiki/topic.md"
        topic.write_text(
            topic.read_text(encoding="utf-8").replace(
                "`src/app.py`", "`scratch/ghost.py`"
            ),
            encoding="utf-8",
        )
        self.assert_fails(
            root, "missing authority path: scratch/ghost.py", bootstrap=False
        )

    def test_staged_file_can_satisfy_ground_truth_authority(self) -> None:
        root = self.make_root(staged=False)
        (root / "src/ghost.py").write_text("GHOST = True\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "src/ghost.py"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        topic = root / "wiki/topic.md"
        topic.write_text(
            topic.read_text(encoding="utf-8").replace(
                "`src/app.py`", "`src/ghost.py`"
            ),
            encoding="utf-8",
        )
        self.assert_passes(root, bootstrap=False)

    def test_expected_generated_output_exemption_is_allowed(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + """

### Expected Generated Outputs

| Expected path | Producer |
| --- | --- |
| `generated/output.bin` | fixture build |
""",
            encoding="utf-8",
        )
        self.assert_passes(root, bootstrap=True)

    def test_missing_path_outside_generated_exemption_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "`src/app.py`", "`generated/output.bin`"
            )
            + """

### Expected Generated Outputs

| Expected path | Producer |
| --- | --- |
| `generated/output.bin` | fixture build |
""",
            encoding="utf-8",
        )
        self.assert_fails(root, "missing authority path: generated/output.bin", bootstrap=True)

    def test_unknown_index_kind_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/index.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "| [Topic](topic.md) | leaf |", "| [Topic](topic.md) | essay |"
            ),
            encoding="utf-8",
        )
        self.assert_fails(root, "unknown Kind: essay", bootstrap=True)

    def test_leaf_missing_required_heading_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/topic.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "## Current Contract", "## Behavior"
            ),
            encoding="utf-8",
        )
        self.assert_fails(root, "leaf missing heading: Current Contract", bootstrap=True)

    def test_fingerprint_assignments_fail(self) -> None:
        assignments = ("source_hash: abc", "sha256=abc", "mtime: 123", "last_verified: today")
        for assignment in assignments:
            with self.subTest(assignment=assignment):
                root = self.make_root(staged=True)
                path = root / "wiki/topic.md"
                path.write_text(
                    path.read_text(encoding="utf-8") + f"\n{assignment}\n",
                    encoding="utf-8",
                )
                self.assert_fails(root, "stored fingerprint metadata", bootstrap=True)

    def test_duplicate_catalog_alias_fails(self) -> None:
        root = self.make_root(staged=True)
        path = root / "wiki/catalog.md"
        path.write_text(
            path.read_text(encoding="utf-8")
            + "| APP BEHAVIOR | [Topic](topic.md) | `src/app.py` |\n",
            encoding="utf-8",
        )
        self.assert_fails(root, "duplicate catalog alias", bootstrap=True)

    def test_unknown_unindexed_page_fails(self) -> None:
        root = self.make_root(staged=True)
        (root / "wiki/orphan.md").write_text("# Orphan\n", encoding="utf-8")
        self.assert_fails(root, "live page missing from index: wiki/orphan.md", bootstrap=True)

    def test_whitespace_and_final_newline_fail(self) -> None:
        for payload, fragment in (("bad trailing   \n", "trailing whitespace"), ("no newline", "missing final newline")):
            with self.subTest(fragment=fragment):
                root = self.make_root(staged=True)
                path = root / "wiki/topic.md"
                path.write_bytes(path.read_bytes() + payload.encode("utf-8"))
                self.assert_fails(root, fragment, bootstrap=True)

    def test_impact_paths_report_overlap_exclusion_and_unmapped(self) -> None:
        root = self.make_root(staged=False)
        impact = root / "wiki/change-impact.md"
        impact.write_text(
            impact.read_text(encoding="utf-8")
            + "| `src/**` | [Source map](source-map.md) | [Topic](topic.md) | Review broad source route |\n",
            encoding="utf-8",
        )
        result = self.run_impact(root, "src/app.py", "plan1.md", "unknown.txt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("path: src/app.py", result.stdout)
        self.assertIn("route: src/*.py", result.stdout)
        self.assertIn("route: src/**", result.stdout)
        self.assertIn("primary: [Topic](topic.md)", result.stdout)
        self.assertIn("primary: [Source map](source-map.md)", result.stdout)
        self.assertIn("also review: [Topic](topic.md)", result.stdout)
        self.assertIn("verification: Unit test", result.stdout)
        self.assertIn("status: excluded", result.stdout)
        self.assertIn("status: unmapped", result.stdout)
        self.assertNotIn("fresh", result.stdout.casefold())
        self.assertNotIn("stale", result.stdout.casefold())

    def test_impact_directory_path_reports_nested_routes(self) -> None:
        root = self.make_root(staged=False)
        result = self.run_impact(root, "src")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("route: src/*.py", result.stdout)

    def test_impact_runs_from_non_root_working_directory(self) -> None:
        root = self.make_root(staged=False)
        result = self.run_impact(root, "src/app.py", cwd=root / "wiki")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("path: src/app.py", result.stdout)

    def test_impact_rejects_malformed_table(self) -> None:
        root = self.make_root(staged=False)
        impact = root / "wiki/change-impact.md"
        impact.write_text(
            impact.read_text(encoding="utf-8").replace(
                "| `src/*.py` | [Topic](topic.md) | None | Unit test |",
                "| `src/*.py` | [Topic](topic.md) | Unit test |",
            ),
            encoding="utf-8",
        )
        result = self.run_impact(root, "src/app.py")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("malformed impact row", result.stdout + result.stderr)

    def test_impact_base_and_head_use_historical_diff_without_log_mutation(self) -> None:
        root = self.make_root(staged=False)
        base = self.commit(root, "base")
        log_before = (root / "wiki/log.md").read_text(encoding="utf-8")
        (root / "src/app.py").write_text("VALUE = 2\n", encoding="utf-8")
        head = self.commit(root, "change app")

        explicit = self.run_impact(root, "--base", base, "--head", head)
        default_head = self.run_impact(root, "--base", base)
        for result in (explicit, default_head):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("path: src/app.py", result.stdout)
            self.assertIn("route: src/*.py", result.stdout)
        self.assertEqual((root / "wiki/log.md").read_text(encoding="utf-8"), log_before)

    def test_impact_requires_base_with_head(self) -> None:
        root = self.make_root(staged=False)
        result = self.run_impact(root, "--head", "HEAD")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--head requires --base", result.stdout + result.stderr)

    def test_impact_rejects_option_shaped_git_endpoint(self) -> None:
        root = self.make_root(staged=False)
        self.commit(root, "base")
        result = self.run_impact(root, "--base=--output=owned")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid Git tree", result.stderr)
        self.assertFalse((root / "owned").exists())

    def test_ci_range_pull_request_and_ordinary_push(self) -> None:
        root = self.make_root(staged=False)
        base = self.commit(root, "base")
        (root / "src/app.py").write_text("VALUE = 2\n", encoding="utf-8")
        head = self.commit(root, "head")

        pull_request = self.run_ci_range(
            root,
            "pull_request",
            {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}},
            head,
        )
        push = self.run_ci_range(
            root,
            "push",
            {"before": base, "after": head, "created": False, "deleted": False},
            head,
        )
        for result in (pull_request, push):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"base={base}", result.stdout)
            self.assertIn(f"head={head}", result.stdout)
            self.assertIn("skip=false", result.stdout)

    def test_ci_range_new_branch_and_unreachable_before_use_default_merge_base(self) -> None:
        root = self.make_root(staged=False)
        base = self.commit(root, "base")
        default_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=root, check=True)
        (root / "src/app.py").write_text("VALUE = 2\n", encoding="utf-8")
        head = self.commit(root, "feature")
        common = {"after": head, "deleted": False, "repository": {"default_branch": default_branch}}

        created = self.run_ci_range(
            root,
            "push",
            {**common, "before": "0" * 40, "created": True},
            head,
        )
        unreachable = self.run_ci_range(
            root,
            "push",
            {**common, "before": "f" * 40, "created": False},
            head,
        )
        for result in (created, unreachable):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"base={base}", result.stdout)
            self.assertIn(f"head={head}", result.stdout)

    def test_ci_range_new_unrelated_branch_falls_back_to_empty_tree(self) -> None:
        root = self.make_root(staged=False)
        self.commit(root, "base")
        default_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        empty_tree = subprocess.run(
            ["git", "hash-object", "-t", "tree", "/dev/null"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        unrelated = subprocess.run(
            ["git", "commit-tree", empty_tree, "-m", "unrelated"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
            env={"GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"},
        ).stdout.strip()
        result = self.run_ci_range(
            root,
            "push",
            {
                "before": "0" * 40,
                "after": unrelated,
                "created": True,
                "deleted": False,
                "repository": {"default_branch": default_branch},
            },
            unrelated,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"base={empty_tree}", result.stdout)
        self.assertIn(f"head={unrelated}", result.stdout)

    def test_ci_range_deleted_push_skips_tree_checks(self) -> None:
        root = self.make_root(staged=False)
        before = self.commit(root, "base")
        result = self.run_ci_range(
            root,
            "push",
            {"before": before, "after": "0" * 40, "created": False, "deleted": True},
            before,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("skip=true", result.stdout)
        self.assertIn("reason=deleted ref has no head tree", result.stdout)

    def test_ci_range_manual_uses_input_or_validated_parent(self) -> None:
        root = self.make_root(staged=False)
        base = self.commit(root, "base")
        (root / "src/app.py").write_text("VALUE = 2\n", encoding="utf-8")
        head = self.commit(root, "head")
        with_input = self.run_ci_range(
            root, "workflow_dispatch", {}, head, "--manual-base", base
        )
        with_parent = self.run_ci_range(root, "workflow_dispatch", {}, head)
        for result in (with_input, with_parent):
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"base={base}", result.stdout)
            self.assertIn(f"head={head}", result.stdout)

    def test_ci_range_manual_without_input_or_parent_fails(self) -> None:
        root = self.make_root(staged=False)
        head = self.commit(root, "only commit")
        result = self.run_ci_range(root, "workflow_dispatch", {}, head)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manual dispatch needs base input because HEAD has no parent", result.stderr)

    def test_disposable_compounding_ingest_updates_existing_owner_and_links(self) -> None:
        root = self.make_root(staged=False)
        self.assert_passes(root, bootstrap=False)
        owner = root / "wiki/topic.md"
        catalog = root / "wiki/catalog.md"
        owner_before = owner.read_text(encoding="utf-8")
        catalog_before = catalog.read_text(encoding="utf-8")

        (root / "src/app.py").write_text("VALUE = 1\nNEW_VALUE = 2\n", encoding="utf-8")
        owner.write_text(
            owner_before.replace(
                "| `src/app.py` | Product source |",
                "| `src/app.py` | Product source for `VALUE` and `NEW_VALUE` |",
            ).replace("Value exists.", "`VALUE` and `NEW_VALUE` exist."),
            encoding="utf-8",
        )
        catalog.write_text(
            catalog_before + "| new value | [Topic](topic.md) | `src/app.py` |\n",
            encoding="utf-8",
        )

        self.assertNotEqual(owner.read_text(encoding="utf-8"), owner_before)
        self.assertNotEqual(catalog.read_text(encoding="utf-8"), catalog_before)
        self.assertIn("[Topic](topic.md)", catalog.read_text(encoding="utf-8"))
        self.assert_passes(root, bootstrap=False)

    def test_repository_workflow_keeps_structure_and_impact_separate(self) -> None:
        workflow = (SCRIPT.parents[1] / ".github/workflows/kb.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pull_request:", workflow)
        self.assertIn("push:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("uses: actions/checkout@v6", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("run: python3 scripts/kb.py check", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn('impact --base "$BASE_SHA" --head "$HEAD_SHA"', workflow)
        self.assertIn('git diff --check "$BASE_SHA" "$HEAD_SHA" --', workflow)
        self.assertNotIn("--bootstrap", workflow)


if __name__ == "__main__":
    unittest.main()
