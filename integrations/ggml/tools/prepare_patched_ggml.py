#!/usr/bin/env python3
"""Create a clean, pinned GGML build tree and apply the vBuf AV backend patch."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--patch-file", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    patch_file = args.patch_file.resolve()
    actual_commit = run("git", "rev-parse", "HEAD", cwd=source)
    if actual_commit != args.commit:
        raise SystemExit(f"GGML source HEAD mismatch: expected {args.commit}, found {actual_commit}")
    if not patch_file.is_file():
        raise SystemExit(f"GGML patch is missing: {patch_file}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    patch_hash = hashlib.sha256(patch_file.read_bytes()).hexdigest()
    marker_text = f"vbuf-generated-ggml\n{args.commit}\n{patch_hash}\n"
    marker = destination / ".vbuf-generated-source"
    if marker.is_file() and marker.read_text(encoding="utf-8") == marker_text:
        print(f"using prepared patched GGML {args.commit} at {destination}")
        return 0

    with tempfile.TemporaryDirectory(prefix="vbuf-ggml-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "source"
        staged.mkdir()
        archive = subprocess.Popen(
            ["git", "archive", "--format=tar", args.commit],
            cwd=source,
            stdout=subprocess.PIPE,
        )
        assert archive.stdout is not None
        with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
            tar.extractall(staged)
        if archive.wait() != 0:
            raise SystemExit("git archive failed while preparing pinned GGML sources")

        subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch_file)],
            cwd=staged,
            check=True,
        )
        (staged / ".vbuf-generated-source").write_text(marker_text, encoding="utf-8")

        if destination.exists():
            marker = destination / ".vbuf-generated-source"
            if not marker.is_file() or marker.read_text(encoding="utf-8").splitlines()[0] != "vbuf-generated-ggml":
                raise SystemExit(f"refusing to replace non-generated GGML directory: {destination}")
            shutil.rmtree(destination)
        os.replace(staged, destination)

    print(f"prepared patched GGML {args.commit} at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
