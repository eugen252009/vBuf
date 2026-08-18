#!/usr/bin/env python3
"""Prepare and verify the canonical pinned llama.cpp Step 21 source tree."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

PINNED = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/llama.cpp/0000-pinned-step21-canonical.patch"
EXPECTED_PATCH_SHA256 = "b365448a51b9e2801d8b86269317e9396a975f19c9562d9534ed34de25e7cb38"


def run(root: Path, *args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def diff_hash(root: Path) -> str:
    # Include the canonical patch's new header even though it is not tracked by
    # the pristine upstream checkout.
    subprocess.run(["git", "-C", str(root), "add", "-N", "src/llama-model-source.h"], check=True)
    diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", PINNED], text=False
    )
    return hashlib.sha256(diff).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve()
    if not (upstream / ".git").exists() and not (upstream / ".git").is_file():
        raise SystemExit(f"not a git checkout: {upstream}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--short"], text=True).strip():
        raise SystemExit("upstream checkout must be clean before preparation")
    if run(upstream, "rev-parse", "HEAD", capture=True) != PINNED:
        raise SystemExit("upstream checkout is not at the pinned commit")
    patch_hash = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    if patch_hash != EXPECTED_PATCH_SHA256:
        raise SystemExit("canonical patch hash does not match its recorded value")
    run(upstream, "apply", "--check", str(PATCH))
    run(upstream, "apply", str(PATCH))
    prepared_hash = diff_hash(upstream)
    if prepared_hash != EXPECTED_PATCH_SHA256:
        raise SystemExit("prepared tree delta does not match the canonical patch")
    print(f"PINNED_LLAMA_COMMIT={PINNED}")
    print(f"CANONICAL_PATCH={PATCH}")
    print(f"PREPARED_TREE_DIFF_SHA256={prepared_hash}")
    print("PREPARED_TREE_MATCHES_CANONICAL_DELTA=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
