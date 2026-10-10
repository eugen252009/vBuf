#!/usr/bin/env python3
"""Generate the chronological index of research Markdown documents.

Historical files use their Git first-addition date only when that addition is
reachable from LEGACY_RESEARCH_CUTOFF. New papers should carry an ISO date in
frontmatter or their path; otherwise they remain in the undated section. This
keeps the index stable before and after a new paper's creating commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass
from urllib.parse import quote, unquote


LEGACY_RESEARCH_CUTOFF = "218ffaa561b3b7a647661af15819b4784a30d1c0"
INDEX_PATH = "research/INDEX.md"
DATE_KEYS = {
    "date",
    "created",
    "created_at",
    "research_date",
    "research-date",
    "published",
}
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:$|[Tt ])")
PATH_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


class IndexErrorDetail(Exception):
    """A user-actionable research-index generation error."""


@dataclass(frozen=True)
class Document:
    path: str
    title: str
    date: dt.date | None
    summary: str | None


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise IndexErrorDetail(f"git {' '.join(args)} failed: {detail}")
    return result


def decode_path(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IndexErrorDetail(
            f"research path is not valid UTF-8 and cannot be linked on GitHub: {raw!r}"
        ) from exc


def staged_documents(root: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    """Read the staged Markdown snapshot and staged rename aliases."""
    listing = git(root, "ls-files", "--stage", "-z", "--", "research").stdout
    documents: dict[str, bytes] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, object_id, stage = header.decode("ascii").split()
        except (ValueError, UnicodeDecodeError) as exc:
            raise IndexErrorDetail(f"cannot parse staged Git entry: {record!r}") from exc
        path = decode_path(raw_path)
        if not path.lower().endswith(".md") or path == INDEX_PATH:
            continue
        if stage != "0":
            raise IndexErrorDetail(
                f"{path}: has unresolved merge stages; resolve it before indexing"
            )
        blob = git(root, "cat-file", "blob", object_id).stdout
        documents[path] = blob

    rename_aliases: dict[str, str] = {}
    changes = git(
        root, "diff", "--cached", "--name-status", "-M", "-z", "--", "research"
    ).stdout.split(b"\0")
    fields = [field for field in changes if field]
    i = 0
    while i < len(fields):
        status = fields[i].decode("ascii", "replace")
        i += 1
        if status.startswith(("R", "C")):
            if i + 1 >= len(fields):
                raise IndexErrorDetail("cannot parse staged rename/copy while indexing")
            old_path, new_path = decode_path(fields[i]), decode_path(fields[i + 1])
            rename_aliases[new_path] = old_path
            i += 2
        else:
            if i >= len(fields):
                raise IndexErrorDetail("cannot parse staged research path change")
            i += 1
    return documents, rename_aliases


def working_documents(root: Path) -> dict[str, bytes]:
    research = root / "research"
    if not research.is_dir():
        raise IndexErrorDetail(f"research directory does not exist: {research}")
    documents: dict[str, bytes] = {}
    for path in research.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        relative = path.relative_to(root).as_posix()
        if relative == INDEX_PATH:
            continue
        try:
            documents[relative] = path.read_bytes()
        except OSError as exc:
            raise IndexErrorDetail(f"cannot read research document {relative}: {exc}") from exc
    return documents


def frontmatter(lines: list[str]) -> tuple[list[str], str | None, str | None]:
    """Return body lines, explicit date value, and explicit summary value."""
    if not lines or lines[0].strip() != "---":
        return lines, None, None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
    if end is None:
        return lines, None, None
    date_value = summary_value = None
    for line in lines[1:end]:
        match = re.match(r"^\s*([A-Za-z_-]+)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key, value = match.group(1).lower(), match.group(2)
        value = value.split(" #", 1)[0].strip().strip("\"'")
        if key in DATE_KEYS and date_value is None:
            date_value = value
        elif key == "summary" and summary_value is None:
            summary_value = value
    return lines[end + 1 :], date_value, summary_value


def extract_title(path: str, body: list[str]) -> str:
    in_fence = False
    for line in body:
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
        if heading:
            return heading.group(1).strip()
    stem = PATH_DATE_RE.sub("", Path(path).stem).strip("-_ .")
    words = re.sub(r"[-_]+", " ", stem).split()
    return " ".join(word[:1].upper() + word[1:] for word in words) or "Untitled Research"


def extract_summary(body: list[str], front_summary: str | None) -> str | None:
    if front_summary:
        candidate = front_summary
    else:
        candidate = ""
        in_fence = False
        in_summary = False
        for line in body:
            fence = re.match(r"^\s*(```+|~~~+)", line)
            if fence:
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", line)
            if heading:
                if in_summary:
                    break
                in_summary = heading.group(1).strip().casefold() == "summary"
                continue
            if not in_summary:
                continue
            if not line.strip():
                if candidate:
                    break
                continue
            if re.match(r"^\s*(?:[-*+]\s|\d+\.\s)", line):
                return None
            candidate = f"{candidate} {line.strip()}".strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if not candidate or len(candidate) > 280:
        return None
    return candidate


def parse_iso_date(value: str, path: str, source: str) -> dt.date | None:
    match = DATE_RE.match(value.strip())
    if not match:
        return None
    try:
        return dt.date.fromisoformat(match.group(1))
    except ValueError as exc:
        raise IndexErrorDetail(f"{path}: invalid ISO date {match.group(1)!r} in {source}") from exc


def git_first_addition(root: Path, path: str, aliases: dict[str, str]) -> tuple[str, dt.date] | None:
    """Return the first-added commit and its committer date, following renames."""
    seen: set[str] = set()
    current = path
    while current not in seen:
        seen.add(current)
        result = git(
            root,
            "log",
            "--follow",
            "--diff-filter=A",
            "--format=%H%x09%cs",
            "--",
            current,
            check=False,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.decode("utf-8", "replace").splitlines() if line]
            if lines:
                commit, raw_date = lines[-1].split("\t", 1)
                try:
                    return commit, dt.date.fromisoformat(raw_date)
                except ValueError as exc:
                    raise IndexErrorDetail(
                        f"{current}: Git first-addition date is invalid: {raw_date!r}"
                    ) from exc
        previous = aliases.get(current)
        if previous is None:
            break
        current = previous
    return None


def resolve_date(
    root: Path,
    path: str,
    explicit_date: str | None,
    aliases: dict[str, str],
    cutoff: str,
    diagnostics: list[str],
) -> dt.date | None:
    if explicit_date:
        result = parse_iso_date(explicit_date, path, "frontmatter")
        if result is not None:
            return result
        diagnostics.append(
            f"{path}: date metadata {explicit_date!r} is not ISO YYYY-MM-DD; trying path/history"
        )

    for match in PATH_DATE_RE.finditer(path):
        try:
            return dt.date.fromisoformat(match.group(1))
        except ValueError:
            diagnostics.append(f"{path}: ignoring invalid path date {match.group(1)!r}")

    first_add = git_first_addition(root, path, aliases)
    if first_add is None:
        return None
    commit, date = first_add
    ancestry = git(root, "merge-base", "--is-ancestor", commit, cutoff, check=False)
    if ancestry.returncode == 0:
        return date
    if ancestry.returncode != 1:
        detail = ancestry.stderr.decode("utf-8", "replace").strip()
        raise IndexErrorDetail(
            f"{path}: cannot verify legacy date ancestry against {cutoff}: {detail}"
        )
    return None


def markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def summary_text(value: str) -> str:
    for char in ("\\", "`", "*", "_", "[", "]"):
        value = value.replace(char, "\\" + char)
    return value


def render_index(root: Path, staged: bool = False) -> tuple[bytes, dict[str, int]]:
    if staged:
        contents, aliases = staged_documents(root)
    else:
        contents = working_documents(root)
        aliases = {}

    cutoff = os.environ.get("VBUF_RESEARCH_INDEX_LEGACY_CUTOFF", LEGACY_RESEARCH_CUTOFF)
    resolved = git(root, "rev-parse", "--verify", f"{cutoff}^{{commit}}", check=False)
    if resolved.returncode != 0:
        raise IndexErrorDetail(
            "legacy research-date cutoff is unavailable; the repository needs its full "
            f"history through {LEGACY_RESEARCH_CUTOFF} (or set "
            "VBUF_RESEARCH_INDEX_LEGACY_CUTOFF for an isolated test repository)"
        )
    cutoff = resolved.stdout.decode("ascii").strip()

    diagnostics: list[str] = []
    documents: list[Document] = []
    for path, raw in contents.items():
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IndexErrorDetail(f"{path}: Markdown document is not valid UTF-8") from exc
        body, explicit_date, front_summary = frontmatter(text.splitlines())
        title = extract_title(path, body)
        date = resolve_date(root, path, explicit_date, aliases, cutoff, diagnostics)
        summary = extract_summary(body, front_summary)
        documents.append(Document(path, title, date, summary))

    for message in diagnostics:
        print(f"warning: {message}", file=sys.stderr)

    def key(doc: Document) -> tuple[object, ...]:
        return (doc.title.casefold(), doc.path.casefold(), doc.path)

    dated = sorted((doc for doc in documents if doc.date is not None), key=lambda doc: (-doc.date.toordinal(), *key(doc)))
    undated = sorted((doc for doc in documents if doc.date is None), key=key)
    lines = [
        "# Research Index",
        "",
        "This index lists the research documents maintained in the vBuf-ML repository.",
        "",
        "It is automatically generated from the Markdown documents inside the `research/` directory.",
        "",
        "Do not edit this index manually.",
        "",
        "The original research documents and their Git history are preserved for reproducibility.",
        "",
        "## Research Papers",
        "",
    ]
    years = sorted({doc.date.year for doc in dated if doc.date is not None}, reverse=True)
    for year in years:
        lines.extend((f"### {year}", ""))
        for doc in dated:
            if doc.date is None or doc.date.year != year:
                continue
            relative = doc.path.removeprefix("research/")
            link = quote(relative, safe="/-._~")
            if unquote(link) != relative or not (root / "research" / relative).is_relative_to(root / "research"):
                raise IndexErrorDetail(f"{doc.path}: cannot create a safe relative link")
            entry = f"- **{doc.date.isoformat()}** — [{markdown_text(doc.title)}]({link})"
            lines.append(entry)
            if doc.summary:
                lines.append(f"  - Summary: {summary_text(doc.summary)}")
        lines.append("")

    if undated:
        lines.extend(("## Undated Documents", ""))
        for doc in undated:
            relative = doc.path.removeprefix("research/")
            link = quote(relative, safe="/-._~")
            lines.append(f"- [{markdown_text(doc.title)}]({link})")
            if doc.summary:
                lines.append(f"  - Summary: {summary_text(doc.summary)}")
        lines.append("")

    output = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    stats = {
        "documents": len(documents),
        "dated": len(dated),
        "undated": len(undated),
        "summaries": sum(doc.summary is not None for doc in documents),
    }
    return output, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write research/INDEX.md")
    mode.add_argument("--check", action="store_true", help="check generated index freshness without writing")
    parser.add_argument("--staged", action="store_true", help="generate from the Git index snapshot (for hooks)")
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.staged and not args.write:
        parser.error("--staged is supported only with --write")

    root = (args.root or Path(__file__).resolve().parents[1]).resolve()
    try:
        generated, stats = render_index(root, staged=args.staged)
        destination = root / INDEX_PATH
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                current = destination.read_bytes()
            except FileNotFoundError:
                current = None
            if current != generated:
                destination.write_bytes(generated)
                action = "Wrote"
            else:
                action = "Unchanged"
            print(
                f"{action} {INDEX_PATH}: {stats['documents']} documents "
                f"({stats['dated']} dated, {stats['undated']} undated)."
            )
            return 0

        try:
            current = destination.read_bytes()
        except FileNotFoundError:
            current = None
        if current != generated:
            print(
                f"{INDEX_PATH} is stale; run python3 scripts/generate-research-index.py --write",
                file=sys.stderr,
            )
            return 1
        print(
            f"Research index is current: {stats['documents']} documents "
            f"({stats['dated']} dated, {stats['undated']} undated)."
        )
        return 0
    except (IndexErrorDetail, OSError) as exc:
        print(f"research index error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
