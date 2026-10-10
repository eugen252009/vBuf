#!/usr/bin/env python3
"""Run three independent bounded Pi sessions against the identity-gated vBuf 8B server."""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
EVIDENCE = Path(__file__).resolve().parents[2]
FIXTURE = EVIDENCE / "fixture"
HARNESS = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/var/tmp/vbuf-pi-qwen3-8b-second-turn-workspaces")
VBUF_BUILD = Path("/var/tmp/vbuf-qwen3-8b-protocol-fix-bundle")
ORIGINAL_RUNNER = ROOT / "research/results/vbuf-ml-integration/pi-agent-qwen3-8b-ab-comparison/raw/harness/run_trial.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen3_8b_original_trial_runner", ORIGINAL_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared sandbox runner: {ORIGINAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FIXTURE = FIXTURE
    module.HARNESS = HARNESS
    module.WORKSPACE_ROOT = WORKSPACE_ROOT
    module.VBUF_BUILD = VBUF_BUILD
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    outcomes = []
    original_argv = sys.argv
    for index in range(1, 4):
        run_id = f"multiturn-vbuf-{index:02d}"
        output = output_root / run_id
        sys.argv = [
            str(ORIGINAL_RUNNER), "--backend", "vbuf", "--run-id", run_id,
            "--output-dir", str(output),
        ]
        try:
            exit_code = runner.main()
        finally:
            sys.argv = original_argv
        result_path = output / "result.json"
        manifest_path = output / "manifest.json"
        if not result_path.is_file() or not manifest_path.is_file():
            outcomes.append({"run_id": run_id, "runner_exit_code": exit_code, "status": "MISSING_RESULT"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        workspace = WORKSPACE_ROOT / run_id / "workspace"
        archive = output / "generated-project"
        for entry in result.get("generated_project_files", []):
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe generated path in trial result: {relative}")
            source = workspace / relative
            data = source.read_bytes()
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"generated file changed since trial capture: {relative}")
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["task_fixture"] = "qwen3-8b-multiturn-tool-cycle-v1"
        manifest["runtime_binary_sha256"] = runner.sha256(VBUF_BUILD / "vbuf_compat_server")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        end = result.get("supervisor_trial_end") or {}
        stats = end.get("agent_stats") or {}
        successful = (
            exit_code == 0 and result.get("bwrap_exit_code") == 0 and
            end.get("agent_exit_code") == 0 and end.get("cap_reason") is None and
            stats.get("tool_call_count", 0) >= 3 and stats.get("turn_count", 0) >= 4
        )
        outcome = {
            "run_id": run_id,
            "runner_exit_code": exit_code,
            "agent_exit_code": end.get("agent_exit_code"),
            "tool_call_count": stats.get("tool_call_count"),
            "turn_count": stats.get("turn_count"),
            "http_exchange_count": stats.get("http_exchange_count"),
            "cap_reason": end.get("cap_reason"),
            "generated_project_files": result.get("generated_project_files"),
            "status": "PASS" if successful else "FAIL",
        }
        outcomes.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
    sys.argv = original_argv
    summary = {
        "schema": "qwen3-8b-multiturn-pi-sessions-v1",
        "required_sessions": 3,
        "minimum_tool_calls_per_session": 3,
        "minimum_turns_per_session": 4,
        "sessions": outcomes,
        "overall_status": "PASS" if len(outcomes) == 3 and all(x["status"] == "PASS" for x in outcomes) else "FAIL",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
