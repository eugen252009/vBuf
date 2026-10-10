from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "generate-research-index.py"
HOOK = ROOT / ".githooks" / "pre-commit"


def run_git(root: Path, *args: str, env: dict[str, str] | None = None, check: bool = True):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["git", *args], cwd=root, env=merged, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=check,
    )


def commit(root: Path, message: str, env: dict[str, str] | None = None) -> str:
    run_git(root, "commit", "-m", message, env=env)
    return run_git(root, "rev-parse", "HEAD").stdout.strip()


def init_repo(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "-q")
    run_git(root, "config", "user.name", "Research Index Test")
    run_git(root, "config", "user.email", "research-index@example.invalid")
    research = root / "research"
    research.mkdir(parents=True, exist_ok=True)
    (research / "legacy-paper.md").write_text("# Legacy Paper\n\nHistorical body.\n")
    run_git(root, "add", "research/legacy-paper.md")
    fixed_date = "2024-01-02T12:00:00+00:00"
    first = commit(
        root,
        "initial legacy research",
        {"GIT_AUTHOR_DATE": fixed_date, "GIT_COMMITTER_DATE": fixed_date},
    )
    return first


def run_generator(root: Path, mode: str, *, staged: bool = False, env: dict[str, str] | None = None):
    command = [sys.executable, str(GENERATOR), "--root", str(root), mode]
    if staged:
        command.append("--staged")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["VBUF_RESEARCH_INDEX_LEGACY_CUTOFF"] = merged.get(
        "VBUF_RESEARCH_INDEX_LEGACY_CUTOFF", run_git(root, "rev-parse", "HEAD").stdout.strip()
    )
    return subprocess.run(command, cwd=root, env=merged, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def install_hook(root: Path, cutoff: str) -> None:
    scripts = root / "scripts"
    hooks = root / ".githooks"
    scripts.mkdir(exist_ok=True)
    hooks.mkdir(exist_ok=True)
    shutil.copy2(GENERATOR, scripts / GENERATOR.name)
    shutil.copy2(HOOK, hooks / "pre-commit")
    (hooks / "pre-commit").chmod(0o755)
    run_git(root, "config", "core.hooksPath", ".githooks")
    # The hook's generator uses this test-specific legacy baseline.
    (root / ".test-cutoff").write_text(cutoff)


def hook_env(root: Path) -> dict[str, str]:
    return {"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": (root / ".test-cutoff").read_text()}


class ResearchIndexTests(unittest.TestCase):
    def test_dates_titles_links_summaries_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cutoff = init_repo(root)

            research = root / "research"
            (research / "2026-08-04-zeta.md").write_text("# Zeta Paper\n")
            (research / "2026-08-04-metadata-date.md").write_text(
                "---\ndate: 2025-04-03\nsummary: Explicit frontmatter summary.\n---\n"
                "# Metadata Date Paper\n\n## Summary\nSection summary is lower priority.\n"
            )
            (research / "2026-08-04-alpha.md").write_text(
                "# Alpha Paper\n\n## Summary\nA clear summary with `code`.\n\n## Findings\nDetails.\n"
            )
            nested = research / "nested"
            nested.mkdir()
            (nested / "2024-02-03-no_heading_report.md").write_text("Legacy report with no heading.\n")
            (nested / "2026-03-01-a [b].md").write_text("# A [title] \\\n\nNo explicit summary.\n")
            (research / "INDEX.md").write_text("# stale manual index\n")

            after_cutoff = research / "legacy-no-date.md"
            after_cutoff.write_text("# No Reliable Date\n")
            run_git(root, "add", "research/legacy-no-date.md")
            later = "2025-01-01T00:00:00+00:00"
            commit(root, "add undated legacy-shaped file", {"GIT_AUTHOR_DATE": later, "GIT_COMMITTER_DATE": later})

            result = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(result.returncode, 0, result.stderr)
            output = (research / "INDEX.md").read_bytes()
            second = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(output, (research / "INDEX.md").read_bytes())

            text = output.decode()
            self.assertIn("**2025-04-03** — [Metadata Date Paper]", text)  # metadata beats filename date
            self.assertIn("**2024-01-02** — [Legacy Paper]", text)  # original Git first-add date
            self.assertIn("**2024-02-03** — [No Heading Report]", text)
            self.assertIn("**2026-03-01** — [A \\[title\\] \\\\]", text)
            self.assertIn("a%20%5Bb%5D.md", text)
            self.assertIn("Summary: Explicit frontmatter summary.", text)
            self.assertIn("Summary: A clear summary with \\`code\\`.", text)
            self.assertNotIn("Section summary is lower priority", text)
            self.assertNotIn("stale manual index", text)
            self.assertIn("## Undated Documents", text)
            self.assertIn("[No Reliable Date](legacy-no-date.md)", text)
            self.assertEqual(text.count("legacy-paper.md"), 1)

            positions = [text.index("[Alpha Paper]"), text.index("[Zeta Paper]")]
            self.assertEqual(positions, sorted(positions))  # same date: title order
            self.assertLess(text.index("[Zeta Paper]"), text.index("[Metadata Date Paper]"))
            links = re.findall(r"\]\(([^)]+)\)", text)
            targets = [research / unquote(link) for link in links]
            self.assertEqual(len(targets), len(set(targets)))
            self.assertTrue(all(target.is_file() for target in targets))
            self.assertEqual(len(links), 7)
            self.assertEqual(run_generator(root, "--check", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff}).returncode, 0)

    def test_worktree_add_delete_and_check_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cutoff = init_repo(root)
            first = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(first.returncode, 0, first.stderr)
            baseline = (root / "research/INDEX.md").read_bytes()
            (root / "research/nested").mkdir()
            (root / "research/nested/2025-06-07-added.md").write_text("# Newly Added\n")
            stale = run_generator(root, "--check", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(stale.returncode, 1)
            write = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(write.returncode, 0, write.stderr)
            self.assertIn(b"[Newly Added](nested/2025-06-07-added.md)", (root / "research/INDEX.md").read_bytes())
            (root / "research/nested/2025-06-07-added.md").unlink()
            deleted = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(deleted.returncode, 0, deleted.stderr)
            self.assertEqual((root / "research/INDEX.md").read_bytes(), baseline)

    def test_staged_hook_safety_rename_delete_failure_and_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cutoff = init_repo(root)
            install_hook(root, cutoff)
            run_git(root, "add", ".githooks/pre-commit", "scripts/generate-research-index.py", ".test-cutoff")
            commit(root, "install test hook", hook_env(root))
            # Commit the initial generated index; index-only changes do not recurse.
            first_index = run_generator(root, "--write", env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(first_index.returncode, 0, first_index.stderr)
            run_git(root, "add", "research/INDEX.md")
            commit(root, "add generated research index", hook_env(root))

            staged = root / "research/2025-03-04-staged-paper.md"
            staged.write_text("# Staged Paper\n")
            run_git(root, "add", "research/2025-03-04-staged-paper.md")
            unrelated = root / "research/2026-01-01-unstaged-paper.md"
            unrelated.write_text("# Unstaged Paper\n")
            unrelated_work = root / "unrelated.txt"
            unrelated_work.write_text("unrelated unstaged work\n")
            commit(root, "add staged research paper", hook_env(root))
            index_text = (root / "research/INDEX.md").read_text()
            self.assertIn("[Staged Paper]", index_text)
            self.assertNotIn("Unstaged Paper", index_text)
            status = set(run_git(root, "status", "--short").stdout.splitlines())
            self.assertEqual(status, {"?? research/2026-01-01-unstaged-paper.md", "?? unrelated.txt"})
            committed_paths = set(run_git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").stdout.splitlines())
            self.assertEqual(committed_paths, {"research/INDEX.md", "research/2025-03-04-staged-paper.md"})

            # A staged rename must retain the original paper's first-addition date.
            run_git(root, "mv", "research/legacy-paper.md", "research/renamed-legacy.md")
            commit(root, "rename legacy paper", hook_env(root))
            index_text = (root / "research/INDEX.md").read_text()
            self.assertIn("**2024-01-02** — [Legacy Paper](renamed-legacy.md)", index_text)
            self.assertNotIn("legacy-paper.md)", index_text)

            run_git(root, "rm", "research/2025-03-04-staged-paper.md")
            commit(root, "delete paper", hook_env(root))
            self.assertNotIn("Staged Paper", (root / "research/INDEX.md").read_text())

            index_before = (root / "research/INDEX.md").read_bytes()
            (root / "README.md").write_text("unrelated tracked file\n")
            run_git(root, "add", "README.md")
            commit(root, "unrelated change", hook_env(root))
            self.assertEqual(index_before, (root / "research/INDEX.md").read_bytes())

            before_failed_commit = run_git(root, "rev-parse", "HEAD").stdout.strip()
            broken = root / "research/2026-05-06-failure.md"
            broken.write_text("# Must Not Commit\n")
            run_git(root, "add", "research/2026-05-06-failure.md")
            failed = run_git(
                root,
                "commit",
                "-m",
                "reject broken index generation",
                env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": "not-a-valid-commit"},
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(run_git(root, "rev-parse", "HEAD").stdout.strip(), before_failed_commit)
            self.assertEqual(run_git(root, "diff", "--cached", "--name-only").stdout.strip(), "research/2026-05-06-failure.md")
            self.assertNotIn("Must Not Commit", (root / "research/INDEX.md").read_text())

    def test_staged_snapshot_does_not_read_unstaged_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cutoff = init_repo(root)
            (root / "research/2025-02-03-paper.md").write_text("# Staged Version\n")
            run_git(root, "add", "research/2025-02-03-paper.md")
            (root / "research/2025-02-03-paper.md").write_text("# Unstaged Version\n")
            result = run_generator(root, "--write", staged=True, env={"VBUF_RESEARCH_INDEX_LEGACY_CUTOFF": cutoff})
            self.assertEqual(result.returncode, 0, result.stderr)
            text = (root / "research/INDEX.md").read_text()
            self.assertIn("Staged Version", text)
            self.assertNotIn("Unstaged Version", text)


if __name__ == "__main__":
    unittest.main()
