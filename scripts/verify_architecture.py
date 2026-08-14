#!/usr/bin/env python3
"""Verify the one-way vbuf-ml -> generic-vBuf dependency boundary."""
from __future__ import annotations

import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "Cargo.toml"


def dependency_names(package: dict) -> set[str]:
    return {dependency["name"] for dependency in package["dependencies"]}


def main() -> None:
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--manifest-path", str(MANIFEST), "--format-version", "1", "--no-deps"],
            cwd=ROOT,
            text=True,
        )
    )
    packages = {package["name"]: package for package in metadata["packages"]}
    required = {"vbuf-core", "vbuf-layout", "vbuf-ml"}
    if not required <= packages.keys():
        raise SystemExit(f"missing workspace packages: {sorted(required - packages.keys())}")

    core_deps = dependency_names(packages["vbuf-core"])
    layout_deps = dependency_names(packages["vbuf-layout"])
    ml_deps = dependency_names(packages["vbuf-ml"])
    if "vbuf-ml" in core_deps or "vbuf-ml" in layout_deps:
        raise SystemExit("generic package depends on vbuf-ml")
    if "vbuf-core" not in ml_deps or "vbuf-layout" not in ml_deps:
        raise SystemExit("vbuf-ml does not depend on the required generic packages")

    for directory in (ROOT / "rust" / "src", ROOT / "rust" / "vbuf-layout"):
        for path in directory.rglob("*.rs"):
            text = path.read_text()
            if "vbuf_ml" in text or "vbuf-ml" in text:
                raise SystemExit(f"reverse ML reference in generic source: {path}")

    print("verified vbuf-ml -> vbuf-core/vbuf-layout dependency direction")


if __name__ == "__main__":
    main()
