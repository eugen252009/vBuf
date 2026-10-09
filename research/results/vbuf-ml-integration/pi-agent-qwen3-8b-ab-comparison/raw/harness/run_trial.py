#!/usr/bin/env python3
"""Run one bounded Pi Snake trial inside a read-only, network-isolated bwrap."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue

ROOT = Path(__file__).resolve().parents[6]
EVIDENCE = Path(__file__).resolve().parents[2]
FIXTURE = EVIDENCE / "benchmark-task-v1"
HARNESS = EVIDENCE / "raw" / "harness"
WORKSPACE_ROOT = Path("/var/tmp/vbuf-pi-qwen3-ab-workspaces")
MODEL_DIR = Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-8b")
GGUF = MODEL_DIR / "Qwen3-8B-Q4_K_M.gguf"
VBUF = MODEL_DIR / "Qwen3-8B-Q4_K_M.vbuf"
SEMANTIC = MODEL_DIR / "Qwen3-8B-Q4_K_M.semantic.vbuf"
LIBVBUF = Path("/var/tmp/vbuf-pi-qwen3-ab-runtime/libvbuf_ml.so")
VBUF_BUILD = Path("/var/tmp/vbuf-qwen3-8b-ab-build")
LLAMA_BUILD = Path("/home/eugen/projekte/llama.cpp/build")
NODE = Path("/home/eugen/.nvm/versions/node/v24.16.0/bin/node")
RANGE_SERVER = ROOT / "scripts" / "range_server.py"

MAX_OUTPUT_TOKENS = 6144
MAX_TURNS = 32
EXPECTED_SHA256 = {
    "gguf": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    "vbuf": "cc85fa7afd90808484485de0e0a09e88ff5b98b58d1fb69c7083f64916417cb5",
    "semantic": "99f2892dbe58d427605457729a6edcfa037d67f8ec1ff7f325953345b5d11212",
}


class TrialFiles:
    def __init__(self, output: Path):
        self.output = output
        self.plain = {
            "agent": output / "agent.jsonl",
            "http": output / "http.jsonl",
            "server": output / "server.jsonl",
            "runner": output / "runner.jsonl",
            "pi_stderr": output / "pi-stderr.jsonl",
            "sandbox_stderr": output / "sandbox.stderr",
        }
        for path in self.plain.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        self.handles = {key: path.open("a", encoding="utf-8") for key, path in self.plain.items()}

    def write(self, key: str, text: str) -> None:
        handle = self.handles[key]
        handle.write(text + "\n")
        handle.flush()

    def close_and_compress(self) -> None:
        for handle in self.handles.values():
            handle.close()
        for key, source in self.plain.items():
            if key == "sandbox_stderr":
                target = self.output / "sandbox.stderr.gz"
            else:
                target = self.output / f"{key}.jsonl.gz"
            with source.open("rb") as src, target.open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as dest:
                    shutil.copyfileobj(src, dest)
            source.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return result.stdout.strip()[:4000]
    except Exception as error:
        return f"ERROR: {error!r}"


def nvidia_snapshot() -> dict:
    query = [
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.run(query, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10,
        )
        return {"gpu_query": gpu.stdout.strip(), "compute_apps": proc.stdout.strip()}
    except Exception as error:
        return {"error": repr(error)}


def prepare_workspace(run_id: str) -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = WORKSPACE_ROOT / run_id
    workspace = run_root / "workspace"
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing independent run directory: {run_root}")
    workspace.mkdir(parents=True, mode=0o700)
    shutil.copyfile(FIXTURE / "README.md", workspace / "README.md")
    (workspace / ".home").mkdir(mode=0o700)
    (workspace / ".pi-agent").mkdir(mode=0o700)
    return workspace


def bwrap_command(backend: str, workspace: Path, template_probe: Path | None = None) -> list[str]:
    command = [
        "bwrap",
        "--unshare-user", "--unshare-pid", "--unshare-net", "--die-with-parent",
        "--ro-bind", "/", "/",
        "--size", "1073741824", "--tmpfs", "/tmp",
        "--dir", "/tmp/runtime-tools",
        "--ro-bind", str(NODE), "/tmp/runtime-tools/node",
        "--ro-bind", str(RANGE_SERVER), "/tmp/runtime-tools/range_server.py",
        "--ro-bind", str(HARNESS / "trial_supervisor.py"), "/tmp/runtime-tools/trial_supervisor.py",
        "--ro-bind", str(HARNESS / "http_proxy.py"), "/tmp/runtime-tools/http_proxy.py",
        "--ro-bind", str(HARNESS / "bash-env.sh"), "/tmp/runtime-tools/bash-env.sh",
        "--ro-bind", str(HARNESS / "qwen3-vbuf-matched.jinja"), "/tmp/runtime-tools/qwen3-vbuf-matched.jinja",
        "--ro-bind", str(FIXTURE / "system-prompt.txt"), "/tmp/runtime-tools/system-prompt.txt",
        "--ro-bind", str(FIXTURE / "pi-user-message.txt"), "/tmp/runtime-tools/pi-user-message.txt",
        "--dir", "/tmp/runtime-libs",
        "--ro-bind", str(LIBVBUF), "/tmp/runtime-libs/libvbuf_ml.so",
    ]
    if template_probe is not None:
        if backend != "llama":
            raise ValueError("template probes are supported only for the llama backend")
        command += ["--ro-bind", str(template_probe), "/tmp/runtime-tools/template-probe-request.json"]

    if backend == "vbuf":
        command += [
            "--dir", "/tmp/vbuf-qwen-native-matrix-build",
            "--ro-bind", str(VBUF_BUILD), "/tmp/vbuf-qwen-native-matrix-build",
        ]
    elif backend == "llama":
        command += [
            "--dir", "/tmp/llama-build",
            "--ro-bind", str(LLAMA_BUILD), "/tmp/llama-build",
        ]
    else:
        raise ValueError(backend)

    command += [
        "--size", "16777216", "--tmpfs", "/home/eugen",
        "--dir", "/home/eugen/.bun",
        "--ro-bind", "/home/eugen/.bun", "/home/eugen/.bun",
        "--dir", "/home/eugen/.cache",
        "--dir", "/home/eugen/.cache/vbuf-agent-qualification",
        "--dir", str(MODEL_DIR),
    ]
    if backend == "llama":
        command += ["--ro-bind", str(GGUF), str(GGUF)]
    else:
        command += [
            "--ro-bind", str(VBUF), str(VBUF),
            "--ro-bind", str(SEMANTIC), str(SEMANTIC),
        ]
    command += [
        "--size", "16777216", "--tmpfs", "/var/tmp",
        "--size", "16777216", "--tmpfs", "/mnt",
        "--bind", str(workspace), "/mnt/workspace",
        "--proc", "/proc",
        "--dev", "/dev",
        "--dev-bind", "/dev/nvidia0", "/dev/nvidia0",
        "--dev-bind", "/dev/nvidiactl", "/dev/nvidiactl",
        "--dev-bind", "/dev/nvidia-uvm", "/dev/nvidia-uvm",
        "--dev-bind", "/dev/nvidia-uvm-tools", "/dev/nvidia-uvm-tools",
        "--clearenv",
        "--setenv", "AB_BACKEND", backend,
        "--setenv", "AB_TEMPLATE_PROBE", "1" if template_probe is not None else "0",
        "--setenv", "HOME", "/mnt/workspace/.home",
        "--setenv", "PATH", "/tmp/runtime-tools:/home/eugen/.bun/bin:/usr/local/bin:/usr/bin:/bin",
        "--setenv", "CUDA_VISIBLE_DEVICES", "0",
        "--setenv", "LD_LIBRARY_PATH", "/tmp/runtime-libs",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "LC_ALL", "C.UTF-8",
        "--chdir", "/mnt/workspace",
        "/usr/bin/python3", "/tmp/runtime-tools/trial_supervisor.py",
    ]
    return command


def workspace_usage(workspace: Path) -> tuple[int, int]:
    total_bytes = 0
    total_files = 0
    for root, _dirs, files in os.walk(workspace):
        for filename in files:
            path = Path(root) / filename
            try:
                total_bytes += path.stat(follow_symlinks=False).st_size
                total_files += 1
            except FileNotFoundError:
                continue
    return total_bytes, total_files


def copy_generated_project(workspace: Path, destination: Path) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".c", ".h", ".md", ".txt", ".sh", ".py", ".json", ".yml", ".yaml", ".mk"}
    copied: list[dict] = []
    for source in sorted(workspace.rglob("*")):
        relative = source.relative_to(workspace)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if source.is_symlink() or not source.is_file():
            continue
        if source.name != "Makefile" and source.suffix.lower() not in allowed_suffixes:
            continue
        if source.stat().st_size > 2 * 1024 * 1024:
            continue
        data = source.read_bytes()
        if b"\0" in data:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied.append({
            "path": relative.as_posix(),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vbuf", "llama"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-probe", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = prepare_workspace(args.run_id)
    files = TrialFiles(output)
    started_at = datetime.now(timezone.utc).isoformat()
    expected = {
        "gguf": GGUF,
        "vbuf": VBUF,
        "semantic": SEMANTIC,
    }
    artifacts = {
        key: {"path": str(path), "sha256": EXPECTED_SHA256[key]}
        for key, path in expected.items()
    }
    manifest = {
        "schema": "vbuf-pi-qwen3-8b-ab-trial-v1",
        "trial_mode": "template_probe_only" if args.template_probe else "agent_trial",
        "run_id": args.run_id,
        "backend": args.backend,
        "started_at_utc": started_at,
        "cwd_inside_sandbox": "/mnt/workspace",
        "workspace_host_path": str(workspace),
        "task_fixture": "benchmark-task-v1",
        "task_readme_sha256": sha256(FIXTURE / "README.md"),
        "pi_user_message_sha256": sha256(FIXTURE / "pi-user-message.txt"),
        "system_prompt_sha256": sha256(FIXTURE / "system-prompt.txt"),
        "artifacts": artifacts,
        "limits": {
            "wall_seconds": 1200,
            "backend_startup_seconds": 180,
            "tool_seconds": 120,
            "tool_calls": 32,
            "turns": 32,
            "max_total_generation_tokens": MAX_TURNS * MAX_OUTPUT_TOKENS,
            "agent_shell_virtual_memory_bytes": 4294967296,
            "agent_shell_cpu_seconds": 120,
            "workspace_max_bytes": 536870912,
            "workspace_max_files": 256,
            "tmpfs_max_bytes": {"tmp": 1073741824, "home": 16777216, "var_tmp": 16777216, "mnt": 16777216},
            "http_exchanges": 40,
            "max_tokens_per_request": MAX_OUTPUT_TOKENS,
            "pi_provider_timeout_seconds": 600,
            "pi_automatic_retries": False,
            "pi_provider_retries": 0,
        },
        "sandbox": {
            "mechanism": "bubblewrap user,pid,network namespaces; read-only root; workspace-only writable bind; only GPU0 device nodes",
            "network": "unshared network namespace; Pi, proxy, backend, and range source use loopback inside the namespace",
            "host_root_read_only": True,
            "workspace_writable": True,
            "model_artifacts_read_only": True,
            "external_repositories_hidden": True,
            "gpu_visible_to_backend": "CUDA_VISIBLE_DEVICES=0; RTX 3060 only",
            "gpu_visible_to_agent_shell": "CUDA_VISIBLE_DEVICES empty",
        },
        "versions": {
            "pi": "1.1.0",
            "node": executable_version([str(NODE), "--version"]),
            "bwrap": executable_version(["bwrap", "--version"]),
            "llama_server": executable_version([str(LLAMA_BUILD / "bin/llama-server"), "--version"]),
            "vbuf_server_sha256": sha256(VBUF_BUILD / "vbuf_compat_server") if (VBUF_BUILD / "vbuf_compat_server").is_file() else None,
            "llama_server_sha256": sha256(LLAMA_BUILD / "bin/llama-server"),
            "vbuf_ml_library_sha256": sha256(LIBVBUF) if LIBVBUF.is_file() else None,
            "node_sha256": sha256(NODE),
            "vbuf_git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, check=True).stdout.strip(),
            "llama_cpp_git_commit": subprocess.run(["git", "-C", "/home/eugen/projekte/llama.cpp", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, check=True).stdout.strip(),
        },
        "nvidia_before": nvidia_snapshot(),
        "llama_prompt_normalization": {
            "mode": "Qwen3 GGUF template derived with the recorded generator; apostrophe escaping and empty thinking suffix normalized to match vBuf renderer",
            "template_sha256": sha256(HARNESS / "qwen3-vbuf-matched.jinja"),
            "template_metadata_sha256": sha256(HARNESS / "qwen3-vbuf-matched-template.json"),
        },
        "expected_llama_server": {
            "context": 12288,
            "parallel_slots": 1,
            "batch": 512,
            "ubatch": 128,
            "gpu_layers": "all",
            "flash_attention": "off",
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1,
            "seed": 42,
            "reasoning": "off",
            "jinja_enable_thinking": False,
        },
        "expected_vbuf_server": {
            "context": 12288,
            "layers": 36,
            "max_new_tokens": MAX_OUTPUT_TOKENS,
            "execution": "experimental exact-identity Qwen3-8B; canonical vBuf runtime",
            "sampling": "greedy argmax; no request-level sampler options",
        },
        "pi_configuration": {
            "provider": "pi-ab-local",
            "model": "qwen3-8b",
            "api": "openai-completions",
            "context_window": 12288,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "provider_request_timeout_ms": 600000,
            "automatic_retries": False,
            "provider_retries": 0,
            "thinking": "off",
            "tools": ["read", "write", "bash"],
            "session": False,
            "extensions": False,
            "mcp": False,
            "skills": False,
            "context_files": False,
            "prompt_templates": False,
            "themes": False,
        },
    }
    for key, path in expected.items():
        actual = sha256(path)
        if actual != EXPECTED_SHA256[key]:
            raise RuntimeError(f"{key} artifact SHA-256 mismatch: {actual}")
        manifest["artifacts"][key]["verified_sha256"] = actual
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    command = bwrap_command(args.backend, workspace, args.template_probe.resolve() if args.template_probe else None)
    (output / "bwrap-command.txt").write_text(" ".join(subprocess.list2cmdline([part]) for part in command) + "\n", encoding="utf-8")
    event_queue: Queue = Queue()

    def read_stdout(pipe) -> None:
        for line in iter(pipe.readline, ""):
            event_queue.put(line.rstrip("\r\n"))
        event_queue.put(None)

    def read_stderr(pipe) -> None:
        with files.plain["sandbox_stderr"].open("a", encoding="utf-8") as handle:
            for line in iter(pipe.readline, ""):
                handle.write(line)
                handle.flush()

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        start_new_session=True,
    )
    threading.Thread(target=read_stdout, args=(process.stdout,), daemon=True).start()
    threading.Thread(target=read_stderr, args=(process.stderr,), daemon=True).start()
    deadline = time.monotonic() + 1200
    stdout_done = False
    end_event = None
    outer_cap_reason = None
    last_usage_check = 0.0
    while not stdout_done:
        now = time.monotonic()
        if now > deadline:
            outer_cap_reason = "outer wall limit 1200s exceeded"
        elif now - last_usage_check >= 1.0:
            last_usage_check = now
            used_bytes, used_files = workspace_usage(workspace)
            if used_bytes > 512 * 1024 * 1024:
                outer_cap_reason = f"workspace size limit exceeded: {used_bytes} bytes"
            elif used_files > 256:
                outer_cap_reason = f"workspace file-count limit exceeded: {used_files} files"
        if outer_cap_reason:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            files.write("runner", json.dumps({"event": "outer_limit", "reason": outer_cap_reason}))
            break
        try:
            line = event_queue.get(timeout=0.5)
        except Empty:
            continue
        if line is None:
            stdout_done = True
            continue
        if line.startswith("@@HTTP "):
            files.write("http", line[len("@@HTTP "):])
        elif line.startswith("@@SERVERLOG "):
            files.write("server", line[len("@@SERVERLOG "):])
        elif line.startswith("@@SUPERVISOR "):
            payload = line[len("@@SUPERVISOR "):]
            files.write("runner", payload)
            try:
                parsed = json.loads(payload)
                if parsed.get("event") == "trial_end":
                    end_event = parsed
            except json.JSONDecodeError:
                pass
        elif line.startswith("@@PIERR "):
            files.write("pi_stderr", line[len("@@PIERR "):])
        else:
            files.write("agent", line)
    try:
        exit_code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
        exit_code = process.wait(timeout=5)
    files.close_and_compress()

    project_files = copy_generated_project(workspace, output / "generated-project")
    result = {
        "run_id": args.run_id,
        "backend": args.backend,
        "bwrap_exit_code": exit_code,
        "outer_cap_reason": outer_cap_reason,
        "supervisor_trial_end": end_event,
        "generated_project_files": project_files,
        "nvidia_after": nvidia_snapshot(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if exit_code == 0 and end_event and end_event.get("agent_exit_code") == 0 and not end_event.get("cap_reason") else 1


if __name__ == "__main__":
    sys.exit(main())
