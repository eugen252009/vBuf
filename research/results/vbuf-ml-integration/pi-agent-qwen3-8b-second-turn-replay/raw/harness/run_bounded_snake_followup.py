#!/usr/bin/env python3
"""Run the original bounded Snake fixture against the patched protocol server."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
ORIGINAL_RUNNER = ROOT / "research/results/vbuf-ml-integration/pi-agent-qwen3-8b-ab-comparison/raw/harness/run_trial.py"
OUTPUT = Path(__file__).resolve().parents[1] / "snake-after-protocol-fix"
WORKSPACE_ROOT = Path("/var/tmp/vbuf-pi-qwen3-8b-snake-followup-workspaces")
VBUF_BUILD = Path("/var/tmp/vbuf-qwen3-8b-protocol-fix-bundle")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("qwen3_8b_snake_followup_runner", ORIGINAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared sandbox runner: {ORIGINAL_RUNNER}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.WORKSPACE_ROOT = WORKSPACE_ROOT
    runner.VBUF_BUILD = VBUF_BUILD
    sys.argv = [
        str(ORIGINAL_RUNNER), "--backend", "vbuf", "--run-id", "snake-after-protocol-fix",
        "--output-dir", str(OUTPUT),
    ]
    exit_code = runner.main()
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["runtime_binary_sha256"] = runner.sha256(VBUF_BUILD / "vbuf_compat_server")
        manifest["qualification_note"] = "Original benchmark-task-v1 prompt and limits; this follow-up is not backfilled into the stopped A/B trials."
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
