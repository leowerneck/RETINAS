#!/usr/bin/env python3
"""Dependency-free checks and routing for the RETINAS Markdown knowledge base.

The parser supports inline Markdown links, reference definitions, raw HTML
href/src attributes, and strict tables documented by AGENTS.md. It does not
attempt full CommonMark parsing; unparsed constructs remain manual-review
territory rather than guessed errors. Impact mappings come only from the strict
human-readable change-impact table.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


KINDS = {"hub", "leaf", "map", "special"}
LOG_OPERATIONS = {
    "bootstrap",
    "ingest",
    "filed-query",
    "lint",
    "curate",
    "resolve",
    "restructure",
}
ROOT_HEADINGS = {
    "Purpose And Scope",
    "Authority And Evidence",
    "Start Here",
    "Task Router",
    "Page Types And Placement",
    "Operations",
    "Write And Safety Policy",
    "Completion Checks",
}
LEAF_HEADINGS = {
    "Purpose",
    "Ground Truth",
    "Current Contract",
    "Verification And Impact",
}
HUB_HEADINGS = {
    "Purpose",
    "Read First",
    "Common Tasks",
    "Ownership Boundary",
}
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
REFERENCE_DEFINITION_RE = re.compile(
    r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*(\S.*)$", re.MULTILINE
)
HTML_LINK_ATTRIBUTE_RE = re.compile(
    r"\b(?:href|src)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))",
    re.IGNORECASE,
)
H1_RE = re.compile(r"^# (.+?)\s*$", re.MULTILINE)
H2_RE = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
CODE_RE = re.compile(r"`([^`\n]+)`")
FINGERPRINT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:source_hash|sha256|mtime|last_verified)\s*[:=]\s*\S",
    re.IGNORECASE | re.MULTILINE,
)
FORBIDDEN_RE = re.compile(
    r"(?:^|/)GRHayL(?:/|$)|(?:^|/)(?:plan[^/]*|tasks[^/]*)\.md$",
    re.IGNORECASE,
)
EXCLUDED_EVIDENCE_TOP_LEVEL = {".git", "GRHayL", "build", "dist", "install"}
EXCLUDED_EVIDENCE_PARTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    line: int
    category: str
    message: str


@dataclass
class Document:
    path: Path
    relative: str
    text: str

    def line_for(self, offset: int) -> int:
        return self.text.count("\n", 0, offset) + 1


@dataclass
class SourceRow:
    pattern: str
    owner: str
    owner_target: Path | None
    line: int


@dataclass
class ImpactRow:
    pattern: str
    owner: str
    owner_target: Path | None
    also_review: str
    verification: str
    line: int


class ImpactTableError(ValueError):
    def __init__(self, line: int, message: str) -> None:
        super().__init__(message)
        self.line = line


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def markdown_tables(document: Document) -> list[tuple[int, list[str], list[tuple[int, list[str]]]]]:
    lines = document.text.splitlines()
    tables: list[tuple[int, list[str], list[tuple[int, list[str]]]]] = []
    index = 0
    while index + 1 < len(lines):
        if lines[index].lstrip().startswith("|") and lines[index + 1].lstrip().startswith("|"):
            header = split_row(lines[index])
            separator = split_row(lines[index + 1])
            if is_separator(separator) and len(header) == len(separator):
                rows: list[tuple[int, list[str]]] = []
                cursor = index + 2
                while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
                    cells = split_row(lines[cursor])
                    if len(cells) == len(header):
                        rows.append((cursor + 1, cells))
                    cursor += 1
                tables.append((index + 1, header, rows))
                index = cursor
                continue
        index += 1
    return tables


def link_target(cell: str, base: Path, root: Path) -> Path | None:
    match = LINK_RE.search(cell)
    if not match:
        return None
    target = clean_link(match.group(1))
    if external_link(target):
        return None
    return (base / target.split("#", 1)[0]).resolve(strict=False)


def clean_link(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    if " " in target:
        target = target.split(" ", 1)[0]
    return target


def external_link(target: str) -> bool:
    return bool(re.match(r"^(?:https?://|mailto:)", target, re.IGNORECASE))


def forbidden_target(target: str) -> bool:
    normalized = target.replace("\\", "/").lstrip("./")
    return bool(FORBIDDEN_RE.search(normalized))


def excluded_evidence_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = Path(normalized).parts
    return (
        forbidden_target(normalized)
        or not parts
        or parts[0] in EXCLUDED_EVIDENCE_TOP_LEVEL
        or any(part in EXCLUDED_EVIDENCE_PARTS for part in parts)
        or normalized in {"Makefile", "meson.build"}
        or normalized.endswith((".pyc", ".pyo"))
    )


def within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def glob_regex(pattern: str) -> re.Pattern[str]:
    output = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 2
                if index < len(pattern) and pattern[index] == "/":
                    output += "(?:.*/)?"
                    index += 1
                else:
                    output += ".*"
                continue
            output += "[^/]*"
        elif char == "?":
            output += "[^/]"
        else:
            output += re.escape(char)
        index += 1
    return re.compile(r"^" + output + r"$")


def matches(pattern: str, relative: str) -> bool:
    return bool(glob_regex(pattern).fullmatch(relative))


def unwrap_code(cell: str) -> str | None:
    match = CODE_RE.fullmatch(cell.strip())
    return match.group(1) if match else None


def parse_strict_impact_table(document: Document, root: Path) -> list[ImpactRow]:
    expected = ["path glob", "primary owner", "also review", "verification"]
    lines = document.text.splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("|")
        and [cell.casefold() for cell in split_row(line)] == expected
    ]
    if len(headers) != 1:
        raise ImpactTableError(
            1,
            "expected exactly one impact table with columns: "
            "Path glob, Primary owner, Also review, Verification",
        )

    header_index = headers[0]
    if header_index + 1 >= len(lines):
        raise ImpactTableError(header_index + 1, "impact table lacks separator row")
    separator = split_row(lines[header_index + 1])
    if len(separator) != len(expected) or not is_separator(separator):
        raise ImpactTableError(header_index + 2, "malformed impact table separator")

    rows: list[ImpactRow] = []
    cursor = header_index + 2
    while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
        cells = split_row(lines[cursor])
        line_number = cursor + 1
        if len(cells) != len(expected):
            raise ImpactTableError(
                line_number,
                f"malformed impact row: expected {len(expected)} cells, found {len(cells)}",
            )
        pattern = unwrap_code(cells[0])
        if not pattern:
            raise ImpactTableError(
                line_number, "malformed impact row: Path glob must be one code path/glob"
            )
        if not cells[1] or not cells[2] or not cells[3]:
            raise ImpactTableError(line_number, "malformed impact row: empty cell")
        rows.append(
            ImpactRow(
                pattern,
                cells[1],
                link_target(cells[1], document.path.parent, root),
                cells[2],
                cells[3],
                line_number,
            )
        )
        cursor += 1
    if not rows:
        raise ImpactTableError(header_index + 1, "impact table contains no routes")
    return rows


def first_non_title_line(document: Document) -> tuple[int, str] | None:
    seen_title = False
    for number, line in enumerate(document.text.splitlines(), 1):
        if not seen_title and line.startswith("# "):
            seen_title = True
            continue
        if seen_title and line.strip():
            return number, line.strip()
    return None


def section_text(document: Document, heading: str) -> list[tuple[int, str]]:
    lines = document.text.splitlines()
    results: list[tuple[int, str]] = []
    start = None
    level = None
    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
        if match and match.group(2) == heading:
            start = index + 1
            level = len(match.group(1))
            continue
        if start is not None:
            if match and len(match.group(1)) <= int(level):
                break
            results.append((index + 1, line))
    return results


class Checker:
    def __init__(self, root: Path, bootstrap: bool) -> None:
        self.root = root.resolve()
        self.bootstrap = bootstrap
        self.errors: list[Diagnostic] = []
        self.warnings: list[Diagnostic] = []
        self.documents: dict[Path, Document] = {}
        self.index_kinds: dict[Path, str] = {}
        self._tracked_paths: list[str] | None = None

    def error(self, document: Document | None, line: int, category: str, message: str) -> None:
        relative = document.relative if document else "."
        self.errors.append(Diagnostic(relative, line, category, message))

    def load(self) -> None:
        candidates = [self.root / "AGENTS.md"]
        wiki = self.root / "wiki"
        if wiki.exists():
            candidates.extend(sorted(wiki.rglob("*.md")))
        for path in candidates:
            if not path.is_file():
                if path.name == "AGENTS.md":
                    self.error(None, 1, "graph", "missing root AGENTS.md")
                continue
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self.error(None, 1, "encoding", f"not UTF-8: {path.relative_to(self.root)}")
                continue
            relative = path.relative_to(self.root).as_posix()
            document = Document(path.resolve(), relative, text)
            self.documents[document.path] = document
            if raw and not raw.endswith(b"\n"):
                self.error(document, len(text.splitlines()), "format", "missing final newline")
            for number, line in enumerate(text.splitlines(), 1):
                if line.endswith((" ", "\t")):
                    self.error(document, number, "format", "trailing whitespace")
            for match in FINGERPRINT_RE.finditer(text):
                self.error(document, document.line_for(match.start()), "policy", "stored fingerprint metadata")

    def check_titles(self) -> None:
        seen: dict[str, Document] = {}
        for document in self.documents.values():
            titles = H1_RE.findall(document.text)
            if len(titles) != 1:
                self.error(document, 1, "contract", "document must contain exactly one H1 title")
                continue
            normalized = titles[0].strip().casefold()
            if normalized in seen:
                self.error(document, 1, "contract", f"duplicate title: {titles[0].strip()}")
            else:
                seen[normalized] = document

    def checked_link_target(
        self, document: Document, raw_target: str, line: int
    ) -> Path | None:
        target = clean_link(raw_target)
        if external_link(target) or target.startswith("#"):
            return None
        if target.startswith(("file:", "/")):
            self.error(document, line, "link", f"absolute local link forbidden: {target}")
            return None
        if forbidden_target(target):
            self.error(document, line, "scope", f"forbidden link target: {target}")
            return None
        path_part = target.split("#", 1)[0]
        resolved = (document.path.parent / path_part).resolve(strict=False)
        if not within(resolved, self.root):
            self.error(document, line, "link", f"link escapes repository: {target}")
            return None
        if not resolved.exists():
            self.error(document, line, "link", f"missing local link target: {target}")
            return None
        return resolved

    def check_links(self) -> dict[Path, set[Path]]:
        adjacency = {path: set() for path in self.documents}
        for document in self.documents.values():
            for match in LINK_RE.finditer(document.text):
                resolved = self.checked_link_target(
                    document, match.group(1), document.line_for(match.start())
                )
                if resolved in self.documents:
                    adjacency[document.path].add(resolved)
            for match in REFERENCE_DEFINITION_RE.finditer(document.text):
                self.checked_link_target(
                    document, match.group(1), document.line_for(match.start())
                )
            for match in HTML_LINK_ATTRIBUTE_RE.finditer(document.text):
                raw_target = next(
                    value for value in match.groups() if value is not None
                )
                self.checked_link_target(
                    document, raw_target, document.line_for(match.start())
                )
        return adjacency

    def check_index(self) -> None:
        path = (self.root / "wiki/index.md").resolve()
        index = self.documents.get(path)
        if index is None:
            self.error(None, 1, "index", "missing wiki/index.md")
            return
        counts: dict[Path, int] = {}
        for _, header, rows in markdown_tables(index):
            if [cell.casefold() for cell in header] != ["page", "kind", "purpose"]:
                continue
            for line, cells in rows:
                target = link_target(cells[0], index.path.parent, self.root)
                kind = cells[1].strip().casefold()
                if kind not in KINDS:
                    self.error(index, line, "index", f"unknown Kind: {cells[1].strip() or '<missing>'}")
                if target is None:
                    self.error(index, line, "index", "page entry requires relative Markdown link")
                    continue
                counts[target] = counts.get(target, 0) + 1
                self.index_kinds[target] = kind
        expected = {path for path in self.documents if path.parent == self.root / "wiki" or within(path, self.root / "wiki")}
        expected.discard(index.path)
        for page in sorted(expected):
            relative = page.relative_to(self.root).as_posix()
            count = counts.get(page, 0)
            if count == 0:
                self.error(index, 1, "index", f"live page missing from index: {relative}")
            elif count > 1:
                self.error(index, 1, "index", f"live page indexed more than once: {relative}")
        for target in sorted(counts):
            if target == index.path:
                self.error(index, 1, "index", "index must not index itself")
            elif target not in expected:
                relative = target.relative_to(self.root).as_posix() if within(target, self.root) else str(target)
                self.error(index, 1, "index", f"index entry is not a live wiki page: {relative}")

    def check_reachability(self, adjacency: dict[Path, set[Path]]) -> None:
        root = (self.root / "AGENTS.md").resolve()
        if root not in adjacency:
            return
        reached: set[Path] = set()
        pending = [root]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(adjacency.get(current, ()))
        for page in sorted(set(self.documents) - reached):
            document = self.documents[page]
            self.error(document, 1, "graph", f"page unreachable from AGENTS.md: {document.relative}")

    def check_heading_contracts(self) -> None:
        root = self.documents.get((self.root / "AGENTS.md").resolve())
        if root:
            headings = set(H2_RE.findall(root.text))
            for required in sorted(ROOT_HEADINGS - headings):
                self.error(root, 1, "contract", f"root missing heading: {required}")
        for path, kind in self.index_kinds.items():
            document = self.documents.get(path)
            if document is None or kind not in KINDS:
                continue
            headings = set(H2_RE.findall(document.text))
            if kind == "leaf":
                for required in sorted(LEAF_HEADINGS - headings):
                    self.error(document, 1, "contract", f"leaf missing heading: {required}")
            elif kind == "hub":
                for required in sorted(HUB_HEADINGS - headings):
                    self.error(document, 1, "contract", f"hub missing heading: {required}")
                if not any(heading.startswith("Evidence Limits") for heading in headings):
                    self.error(document, 1, "contract", "hub missing evidence-limits heading")
            elif kind == "map":
                if not any(heading.startswith("Purpose") for heading in headings):
                    self.error(document, 1, "contract", "map missing purpose heading")
                if not markdown_tables(document):
                    self.error(document, 1, "contract", "map requires strict table")

    def check_catalog(self) -> None:
        document = self.documents.get((self.root / "wiki/catalog.md").resolve())
        if document is None:
            return
        aliases: dict[str, int] = {}
        for _, header, rows in markdown_tables(document):
            if not header or "alias" not in header[0].casefold():
                continue
            for line, cells in rows:
                alias = re.sub(r"[`*_]", "", cells[0]).strip().casefold()
                if alias in aliases:
                    self.error(document, line, "catalog", f"duplicate catalog alias: {cells[0].strip()}")
                else:
                    aliases[alias] = line

    def check_log(self) -> None:
        document = self.documents.get((self.root / "wiki/log.md").resolve())
        if document is None:
            return
        pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}) \| ([a-z-]+) \| .+$")
        for match in H2_RE.finditer(document.text):
            value = match.group(1)
            parsed = pattern.fullmatch(value)
            line = document.line_for(match.start())
            if not parsed:
                self.error(document, line, "log", f"invalid log heading: {value}")
            elif parsed.group(2) not in LOG_OPERATIONS:
                self.error(document, line, "log", f"unknown log operation: {parsed.group(2)}")

    def parse_source_rows(self) -> tuple[str | None, list[SourceRow]]:
        document = self.documents.get((self.root / "wiki/source-map.md").resolve())
        if document is None:
            self.error(None, 1, "source-map", "missing wiki/source-map.md")
            return None, []
        metadata = first_non_title_line(document)
        marker = None
        if metadata and metadata[1] in {
            "KB ownership status: staged",
            "KB ownership status: complete",
        }:
            marker = metadata[1].rsplit(" ", 1)[-1]
        else:
            line = metadata[0] if metadata else 1
            self.error(document, line, "source-map", "missing or invalid ownership status marker")
        rows: list[SourceRow] = []
        expected_header = ["source family", "observed role", "primary owner", "exact evidence"]
        for _, header, table_rows in markdown_tables(document):
            if [cell.casefold() for cell in header] != expected_header:
                continue
            for line, cells in table_rows:
                pattern = unwrap_code(cells[0])
                if not pattern:
                    self.error(document, line, "source-map", "source family must be one code path/glob")
                    continue
                owner = cells[2]
                target = link_target(owner, document.path.parent, self.root)
                rows.append(SourceRow(pattern, owner, target, line))
        if not rows:
            self.error(document, 1, "source-map", "missing strict source inventory table")
        required = "staged" if self.bootstrap else "complete"
        if marker and marker != required:
            self.error(document, metadata[0] if metadata else 1, "lifecycle", f"check mode requires {required} ownership status")
        return marker, rows

    def parse_impact_rows(self) -> list[ImpactRow]:
        document = self.documents.get((self.root / "wiki/change-impact.md").resolve())
        if document is None:
            self.error(None, 1, "impact", "missing wiki/change-impact.md")
            return []
        try:
            return parse_strict_impact_table(document, self.root)
        except ImpactTableError as error:
            self.error(document, error.line, "impact", str(error))
            return []

    def tracked_paths(self) -> list[str]:
        if self._tracked_paths is not None:
            return self._tracked_paths
        result = subprocess.run(
            ["git", "-C", str(self.root), "ls-files"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            self.warnings.append(Diagnostic(".", 1, "git", "git ls-files unavailable; tracked coverage not checked"))
            self._tracked_paths = []
        else:
            self._tracked_paths = sorted(
                line for line in result.stdout.splitlines() if line
            )
        return self._tracked_paths

    def owner_is_unassigned(self, owner: str) -> bool:
        return owner.strip().strip("`").casefold() == "unassigned"

    def check_maps(self, marker: str | None, source_rows: list[SourceRow], impact_rows: list[ImpactRow]) -> None:
        source_doc = self.documents.get((self.root / "wiki/source-map.md").resolve())
        impact_doc = self.documents.get((self.root / "wiki/change-impact.md").resolve())
        tracked = self.tracked_paths()
        live_required = [document.relative for document in self.documents.values()]
        for extra in ("scripts/kb.py", "scripts/tests/test_kb.py"):
            if (self.root / extra).exists():
                live_required.append(extra)
        for relative in sorted(set(tracked + live_required)):
            matching = [row for row in source_rows if matches(row.pattern, relative)]
            if not matching:
                self.error(source_doc, 1, "source-map", f"unmapped tracked path: {relative}")
            elif len(matching) > 1:
                self.error(source_doc, matching[1].line, "source-map", f"multiple primary source families match: {relative}")
        impact_by_pattern = {row.pattern: row for row in impact_rows}
        for row in source_rows:
            unassigned = self.owner_is_unassigned(row.owner)
            if not self.bootstrap and unassigned:
                self.error(source_doc, row.line, "lifecycle", f"complete source map contains unassigned owner: {row.pattern}")
            if not unassigned:
                if row.owner_target is None:
                    self.error(source_doc, row.line, "source-map", f"assigned owner must be a link: {row.pattern}")
                elif row.owner_target not in self.documents:
                    self.error(source_doc, row.line, "source-map", f"owner target is not a live graph page: {row.pattern}")
                elif row.owner_target not in self.index_kinds and row.owner_target not in {
                    (self.root / "AGENTS.md").resolve(),
                    (self.root / "wiki/index.md").resolve(),
                }:
                    self.error(source_doc, row.line, "source-map", f"owner target is not indexed: {row.pattern}")
            impact = impact_by_pattern.get(row.pattern)
            if impact is None and (not self.bootstrap or not unassigned):
                self.error(impact_doc, 1, "impact", f"missing impact route for source family: {row.pattern}")
            elif impact is not None and not unassigned and impact.owner_target != row.owner_target:
                self.error(impact_doc, impact.line, "impact", f"impact primary owner disagrees with source map: {row.pattern}")
        if not self.bootstrap and marker == "complete":
            source_patterns = {row.pattern for row in source_rows}
            for row in impact_rows:
                if row.pattern not in source_patterns:
                    self.error(impact_doc, row.line, "impact", f"impact path lacks source family: {row.pattern}")

    def check_authority_paths(self, source_rows: list[SourceRow], impact_rows: list[ImpactRow]) -> None:
        indexed = [
            relative
            for relative in self.tracked_paths()
            if not excluded_evidence_path(relative)
            and (self.root / relative).is_file()
        ]
        available = set(indexed)
        if "AGENTS.md" not in available:
            available.update(
                document.relative for document in self.documents.values()
            )
            for relative in (
                "scripts/kb.py",
                "scripts/tests/test_kb.py",
                ".github/workflows/kb.yml",
            ):
                if (self.root / relative).is_file():
                    available.add(relative)
        candidates: list[tuple[Document, int, str]] = []
        for document in self.documents.values():
            for line_number, line in section_text(document, "Ground Truth"):
                candidates.extend((document, line_number, value) for value in CODE_RE.findall(line))
        source_doc = self.documents.get((self.root / "wiki/source-map.md").resolve())
        if source_doc:
            for _, header, rows in markdown_tables(source_doc):
                if [cell.casefold() for cell in header] == ["source family", "observed role", "primary owner", "exact evidence"]:
                    for line, cells in rows:
                        candidates.extend((source_doc, line, value) for value in CODE_RE.findall(cells[3]))
        impact_doc = self.documents.get((self.root / "wiki/change-impact.md").resolve())
        if impact_doc:
            candidates.extend((impact_doc, row.line, row.pattern) for row in impact_rows)

        seen: set[tuple[str, int, str]] = set()
        for document, line, target in candidates:
            key = (document.relative, line, target)
            if key in seen:
                continue
            seen.add(key)
            normalized = target.replace("\\", "/").rstrip("/")
            if forbidden_target(normalized):
                self.error(document, line, "scope", f"forbidden authority target: {target}")
                continue
            if not self.looks_like_path(normalized):
                continue
            if any(matches(normalized, path) for path in available):
                continue
            self.error(document, line, "authority", f"missing authority path: {target}")

    @staticmethod
    def looks_like_path(value: str) -> bool:
        return "/" in value or value in {
            "AGENTS.md",
            "README.md",
            "configure",
            "meson.build.in",
            "meson_options.txt",
        }

    def run(self) -> tuple[list[Diagnostic], list[Diagnostic]]:
        self.load()
        self.check_titles()
        adjacency = self.check_links()
        self.check_index()
        self.check_reachability(adjacency)
        self.check_heading_contracts()
        self.check_catalog()
        self.check_log()
        marker, source_rows = self.parse_source_rows()
        impact_rows = self.parse_impact_rows()
        self.check_maps(marker, source_rows, impact_rows)
        self.check_authority_paths(source_rows, impact_rows)
        return sorted(set(self.errors)), sorted(set(self.warnings))


def command_check(args: argparse.Namespace) -> int:
    checker = Checker(Path(args.root), args.bootstrap)
    errors, warnings = checker.run()
    for diagnostic in warnings:
        print(f"{diagnostic.path}:{diagnostic.line}: warning/{diagnostic.category}: {diagnostic.message}")
    for diagnostic in errors:
        print(f"{diagnostic.path}:{diagnostic.line}: {diagnostic.category}: {diagnostic.message}")
    if errors:
        print(f"KB check failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"KB check passed: {len(checker.documents)} graph node(s), {len(warnings)} warning(s)")
    return 0


def repository_root(value: str) -> Path:
    return Path(value).resolve()


def impact_document(root: Path) -> Document:
    path = root / "wiki/change-impact.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ImpactTableError(1, f"cannot read wiki/change-impact.md: {error}") from error
    return Document(path.resolve(), "wiki/change-impact.md", text)


def normalized_repository_path(raw: str, root: Path) -> str:
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve(strict=False).relative_to(root).as_posix()
        except ValueError:
            return candidate.as_posix()
    normalized = raw.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/") or "."


def route_matches_path(pattern: str, path: str) -> bool:
    if matches(pattern, path):
        return True
    prefix = path.rstrip("/") + "/"
    return pattern.startswith(prefix)


def diff_paths(root: Path, base: str, head: str) -> list[str]:
    base_tree = git_output(root, "rev-parse", "--verify", "--end-of-options", f"{base}^{{tree}}")
    head_tree = git_output(root, "rev-parse", "--verify", "--end-of-options", f"{head}^{{tree}}")
    if not base_tree:
        raise ValueError(f"invalid Git tree: {base}")
    if not head_tree:
        raise ValueError(f"invalid Git tree: {head}")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            base_tree,
            head_tree,
            "--",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git diff failed for {base_tree}..{head_tree}: {message}")
    return sorted(
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    )


def command_impact(args: argparse.Namespace) -> int:
    root = repository_root(args.root)
    if args.head and not args.base:
        print("impact: --head requires --base", file=sys.stderr)
        return 2
    if args.base and args.paths:
        print("impact: changed paths and --base are mutually exclusive", file=sys.stderr)
        return 2
    try:
        routes = parse_strict_impact_table(impact_document(root), root)
        paths = (
            diff_paths(root, args.base, args.head or "HEAD")
            if args.base
            else [normalized_repository_path(path, root) for path in args.paths]
        )
    except ImpactTableError as error:
        print(f"wiki/change-impact.md:{error.line}: impact: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"impact: {error}", file=sys.stderr)
        return 1
    if not paths:
        print("Impact review (advisory routing; reopen exact evidence before material claims)")
        print("status: no changed paths")
        return 0

    print("Impact review (advisory routing; reopen exact evidence before material claims)")
    for path in paths:
        print(f"path: {path}")
        if forbidden_target(path):
            print("  status: excluded")
            continue
        matching = [route for route in routes if route_matches_path(route.pattern, path)]
        if not matching:
            print("  status: unmapped")
            continue
        print("  status: routed")
        for route in matching:
            print(f"  route: {route.pattern}")
            print(f"    primary: {route.owner}")
            print(f"    also review: {route.also_review}")
            print(f"    verification: {route.verification}")
    return 0


def git_output(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def commit_sha(root: Path, reference: str) -> str | None:
    return git_output(
        root, "rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}"
    )


def empty_tree_sha(root: Path) -> str:
    value = git_output(root, "hash-object", "-t", "tree", "/dev/null")
    if not value:
        raise ValueError("cannot resolve Git empty tree")
    return value


def all_zero_sha(value: object) -> bool:
    text = str(value or "")
    return len(text) >= 40 and set(text) == {"0"}


def default_branch_commit(root: Path, branch: object) -> str:
    name = str(branch or "")
    if not name:
        raise ValueError("push event lacks repository.default_branch")
    valid = subprocess.run(
        ["git", "check-ref-format", "--branch", name],
        text=True,
        capture_output=True,
        check=False,
    )
    if valid.returncode:
        raise ValueError(f"invalid repository.default_branch: {name}")
    for reference in (f"refs/remotes/origin/{name}", f"refs/heads/{name}", name):
        resolved = commit_sha(root, reference)
        if resolved:
            return resolved
    fetch = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "fetch",
            "--no-tags",
            "origin",
            f"refs/heads/{name}:refs/remotes/origin/{name}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if fetch.returncode == 0:
        resolved = commit_sha(root, f"refs/remotes/origin/{name}")
        if resolved:
            return resolved
    raise ValueError(f"default branch is unavailable after fetch: {name}")


def fallback_push_base(root: Path, payload: dict[str, object], head: str) -> str:
    repository = payload.get("repository")
    default_name = repository.get("default_branch") if isinstance(repository, dict) else None
    default = default_branch_commit(root, default_name)
    merge_base = git_output(root, "merge-base", default, head)
    return merge_base or empty_tree_sha(root)


def resolve_ci_range(
    root: Path,
    event_name: str,
    payload: dict[str, object],
    sha: str,
    manual_base: str | None,
) -> tuple[str | None, str | None, bool, str]:
    if event_name == "pull_request":
        pull_request = payload.get("pull_request")
        if not isinstance(pull_request, dict):
            raise ValueError("pull_request event lacks pull_request object")
        base_data = pull_request.get("base")
        head_data = pull_request.get("head")
        base_ref = base_data.get("sha") if isinstance(base_data, dict) else None
        head_ref = head_data.get("sha") if isinstance(head_data, dict) else None
        base = commit_sha(root, str(base_ref or ""))
        head = commit_sha(root, str(head_ref or ""))
        if not base or not head:
            raise ValueError("pull_request base/head commits are unavailable after full checkout")
        return base, head, False, "pull request base/head"

    if event_name == "push":
        if payload.get("deleted") is True or all_zero_sha(payload.get("after")):
            return None, None, True, "deleted ref has no head tree"
        head_ref = str(payload.get("after") or sha)
        head = commit_sha(root, head_ref)
        if not head:
            raise ValueError(f"push head commit is unavailable: {head_ref}")
        before_ref = str(payload.get("before") or "")
        if payload.get("created") is not True and not all_zero_sha(before_ref):
            before = commit_sha(root, before_ref)
            if before:
                return before, head, False, "ordinary push before/after"
        return fallback_push_base(root, payload, head), head, False, "default-branch merge base"

    if event_name == "workflow_dispatch":
        head = commit_sha(root, sha)
        if not head:
            raise ValueError(f"manual dispatch head commit is unavailable: {sha}")
        if manual_base:
            base = commit_sha(root, manual_base)
            if not base:
                raise ValueError(f"manual dispatch base is not a commit: {manual_base}")
            return base, head, False, "manual base input"
        parent = commit_sha(root, f"{head}^")
        if not parent:
            raise ValueError("manual dispatch needs base input because HEAD has no parent")
        return parent, head, False, "validated HEAD parent"

    raise ValueError(f"unsupported CI event: {event_name}")


def command_ci_range(args: argparse.Namespace) -> int:
    root = repository_root(args.root)
    try:
        payload = json.loads(Path(args.event_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a JSON object")
        base, head, skip, reason = resolve_ci_range(
            root, args.event_name, payload, args.sha, args.manual_base
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f"ci-range: {error}", file=sys.stderr)
        return 1
    print(f"base={base or ''}")
    print(f"head={head or ''}")
    print(f"skip={'true' if skip else 'false'}")
    print(f"reason={reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RETINAS knowledge-base tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate the Markdown knowledge base")
    default_root = str(Path(__file__).resolve().parents[1])
    check.add_argument("--root", default=default_root, help="repository root")
    check.add_argument(
        "--bootstrap",
        action="store_true",
        help="require staged ownership and allow explicit incomplete product rows",
    )
    check.set_defaults(function=command_check)

    impact = subparsers.add_parser("impact", help="print advisory review routes")
    impact.add_argument("--root", default=default_root, help="repository root")
    impact.add_argument("--base", help="Git base ref used to discover changed paths")
    impact.add_argument("--head", help="Git head ref (default: HEAD; requires --base)")
    impact.add_argument("paths", nargs="*", metavar="PATH", help="repository-relative changed path")
    impact.set_defaults(function=command_impact)

    ci_range = subparsers.add_parser(
        "ci-range", help="resolve explicit Git endpoints for KB CI"
    )
    ci_range.add_argument("--root", default=default_root, help="repository root")
    ci_range.add_argument("--event-name", required=True)
    ci_range.add_argument("--event-file", required=True)
    ci_range.add_argument("--sha", required=True)
    ci_range.add_argument("--manual-base")
    ci_range.set_defaults(function=command_ci_range)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
