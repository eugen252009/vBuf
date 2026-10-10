#!/usr/bin/env python3
"""Start one isolated model backend, capture Pi traffic, and enforce trial limits."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from http_proxy import start_proxy

MODEL_ROOT = Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-8b")
GGUF = MODEL_ROOT / "Qwen3-8B-Q4_K_M.gguf"
VBUF = MODEL_ROOT / "Qwen3-8B-Q4_K_M.vbuf"
SEMANTIC = MODEL_ROOT / "Qwen3-8B-Q4_K_M.semantic.vbuf"
BACKEND_PORT = 18101
PROXY_PORT = 18180
RANGE_PORT = 18318
MAX_WALL_SECONDS = 1200
MAX_STARTUP_SECONDS = 180
MAX_TOOL_SECONDS = 120
MAX_TOOL_CALLS = 32
MAX_TURNS = 32
MAX_HTTP_EXCHANGES = 40
MAX_OUTPUT_TOKENS = 6144

SYSTEM_PROMPT = Path("/tmp/runtime-tools/system-prompt.txt").read_text(encoding="utf-8").rstrip("\n")
USER_PROMPT = Path("/tmp/runtime-tools/pi-user-message.txt").read_text(encoding="utf-8").rstrip("\n")


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def drain_stream(stream, label: str, stream_name: str, events: queue.Queue) -> None:
    if stream is not None:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if label == "pi":
                events.put(("pi_" + stream_name, line))
            elif is_relevant_server_line(label, line):
                events.put(("server", {"backend": label, "stream": stream_name, "line": line}))
    events.put((label + "_" + stream_name + "_eof", None))


def drain_process_output(proc: subprocess.Popen, label: str, events: queue.Queue) -> None:
    for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        threading.Thread(target=drain_stream, args=(stream, label, stream_name, events), daemon=True).start()


def is_relevant_server_line(backend: str, line: str) -> bool:
    lower = line.lower()
    if backend == "vbuf":
        return (
            "vbuf_request " in line
            or "vbuf_control_request " in line
            or "listening" in lower
            or "admitted" in lower
            or "error" in lower
            or "failed" in lower
            or "cuda" in lower
        )
    return bool(re.search(r"(listening|loaded|loading|error|failed|srv|slot|prompt eval|eval time|tokens_predicted)", lower))


def wait_health(events: queue.Queue, server: subprocess.Popen) -> float:
    start = time.monotonic()
    url = f"http://127.0.0.1:{BACKEND_PORT}/health"
    while time.monotonic() - start < MAX_STARTUP_SECONDS:
        if server.poll() is not None:
            raise RuntimeError(f"backend exited during startup with status {server.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - start
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"backend failed health check within {MAX_STARTUP_SECONDS}s")


def backend_command(backend: str) -> tuple[list[str], dict[str, str], list[str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    env["OMP_NUM_THREADS"] = "8"
    if backend == "vbuf":
        env["LD_LIBRARY_PATH"] = (
            "/tmp/runtime-libs:/tmp/vbuf-qwen-native-matrix-build/ggml/src:"
            "/tmp/vbuf-qwen-native-matrix-build/ggml/src/ggml-cuda"
        )
        env["VBUF_QUALIFICATION_DUMP_TOKEN_IDS"] = "1"
        range_command = [
            "/usr/bin/python3", "/tmp/runtime-tools/range_server.py",
            "--file", str(VBUF), "--host", "127.0.0.1", "--port", str(RANGE_PORT),
        ]
        command = [
            "/tmp/vbuf-qwen-native-matrix-build/vbuf_compat_server",
            "--semantic-model", str(SEMANTIC),
            "--source-url", f"http://127.0.0.1:{RANGE_PORT}/Qwen3-8B-Q4_K_M.vbuf",
            "--model-alias", "qwen3-8b",
            "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
            "--blocks", "36",
            "--experimental-qwen3-8b",
            "--qwen-context-capacity", "12288",
            "--max-new-tokens", str(MAX_OUTPUT_TOKENS),
        ]
        return command, env, range_command
    if backend == "llama":
        env["LD_LIBRARY_PATH"] = "/tmp/llama-build/bin"
        command = [
            "/tmp/llama-build/bin/llama-server",
            "--model", str(GGUF),
            "--alias", "qwen3-8b",
            "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
            "--ctx-size", "12288",
            "--batch-size", "512", "--ubatch-size", "128",
            "--gpu-layers", "all", "--split-mode", "none", "--main-gpu", "0",
            "--parallel", "1", "--fit", "off", "--flash-attn", "off",
            "--temp", "0", "--top-k", "1", "--top-p", "1",
            "--repeat-penalty", "1", "--seed", "42",
            "--jinja", "--reasoning", "off",
            "--chat-template-file", "/tmp/runtime-tools/qwen3-vbuf-matched.jinja",
            "--no-webui", "--log-verbosity", "3",
        ]
        return command, env, []
    raise ValueError(f"unknown backend {backend!r}")


def prepare_agent_config(workspace: Path) -> None:
    agent_dir = workspace / ".pi-agent"
    agent_dir.mkdir(mode=0o700, exist_ok=True)
    (workspace / ".home").mkdir(mode=0o700, exist_ok=True)
    models = {
        "providers": {
            "pi-ab-local": {
                "baseUrl": f"http://127.0.0.1:{PROXY_PORT}/v1",
                "apiKey": "qwen3-ab-local-only",
                "api": "openai-completions",
                "models": [{
                    "id": "qwen3-8b",
                    "name": "Qwen3-8B local coding benchmark",
                    "reasoning": False,
                    "input": ["text"],
                    "contextWindow": 12288,
                    "maxTokens": MAX_OUTPUT_TOKENS,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "compat": {"supportsStore": False, "supportsUsageInStreaming": False},
                }],
            },
        },
    }
    settings = {
        "defaultThinkingLevel": "off",
        "quietStartup": True,
        "httpIdleTimeoutMs": 600000,
        "retry": {
            "enabled": False,
            "maxRetries": 0,
            "provider": {"timeoutMs": 600000, "maxRetries": 0},
        },
    }
    (agent_dir / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
    (agent_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def run_agent(workspace: Path, events: queue.Queue) -> tuple[int, str | None, dict, list[dict]]:
    env = {
        "HOME": "/mnt/workspace/.home",
        "PATH": "/tmp/runtime-tools:/home/eugen/.bun/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "CI": "1",
        "TMPDIR": "/tmp/pi-temp",
        "PI_CODING_AGENT_DIR": "/mnt/workspace/.pi-agent",
        "PI_CODING_AGENT_SESSION_DIR": "/tmp/pi-sessions",
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        "BASH_ENV": "/tmp/runtime-tools/bash-env.sh",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "npm_config_offline": "true",
        "PIP_NO_INDEX": "1",
    }
    command = [
        "/home/eugen/.bun/bin/pi",
        "--provider", "pi-ab-local", "--model", "qwen3-8b",
        "--mode", "json", "--print", "--offline",
        "--no-session", "--no-mcp", "--no-extensions", "--no-skills",
        "--no-context-files", "--no-prompt-templates", "--no-themes", "--no-approve",
        "--thinking", "off", "--tools", "read,write,bash",
        "--system-prompt", SYSTEM_PROMPT,
        "--", USER_PROMPT,
    ]
    start = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd="/mnt/workspace",
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=set_agent_limits,
    )
    threading.Thread(target=drain_process_output, args=(proc, "pi", events), daemon=True).start()

    tool_started: dict[str, float] = {}
    tool_count = 0
    turn_count = 0
    http_count = 0
    cap_reason = None
    eof_streams: set[str] = set()
    captured_requests: list[dict] = []
    while True:
        finished = proc.poll() is not None and {"pi_stdout_eof", "pi_stderr_eof"}.issubset(eof_streams)
        if finished:
            break
        now = time.monotonic()
        if proc.poll() is None and now - start > MAX_WALL_SECONDS:
            cap_reason = f"run wall limit {MAX_WALL_SECONDS}s exceeded"
            break
        expired = [key for key, started in tool_started.items() if now - started > MAX_TOOL_SECONDS]
        if proc.poll() is None and expired:
            cap_reason = f"tool wall limit {MAX_TOOL_SECONDS}s exceeded for {expired}"
            break
        try:
            kind, value = events.get(timeout=0.25)
        except queue.Empty:
            continue
        if kind == "pi_stdout":
            print(value, flush=True)
            try:
                item = json.loads(value)
            except json.JSONDecodeError:
                item = None
            if isinstance(item, dict):
                event_type = item.get("type")
                if event_type == "turn_start":
                    turn_count += 1
                    if turn_count > MAX_TURNS:
                        cap_reason = f"turn limit {MAX_TURNS} exceeded"
                        break
                elif event_type == "tool_execution_start":
                    tool_count += 1
                    tool_started[str(item.get("toolCallId", tool_count))] = time.monotonic()
                    if tool_count > MAX_TOOL_CALLS:
                        cap_reason = f"tool-call limit {MAX_TOOL_CALLS} exceeded"
                        break
                elif event_type == "tool_execution_end":
                    tool_started.pop(str(item.get("toolCallId", "")), None)
        elif kind == "pi_stderr":
            emit("@@PIERR ", {"line": value})
        elif kind == "server":
            emit("@@SERVERLOG ", value)
        elif kind == "http_exchange":
            emit("@@HTTP ", value)
            if value.get("method") == "POST" and value.get("path") == "/v1/chat/completions":
                http_count += 1
                captured_requests.append({"id": value.get("id"), "body": value.get("request_body")})
                if http_count > MAX_HTTP_EXCHANGES:
                    cap_reason = f"HTTP exchange limit {MAX_HTTP_EXCHANGES} exceeded"
                    break
        elif kind == "http_proxy_error":
            emit("@@HTTP ", value)
        elif kind.endswith("_eof"):
            eof_streams.add(kind)

    if cap_reason:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        exit_code = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        exit_code = proc.wait(timeout=5)
    return exit_code, cap_reason, {
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "turn_count": turn_count,
        "tool_call_count": tool_count,
        "http_exchange_count": http_count,
        "cap_reason": cap_reason,
    }, captured_requests


def set_agent_limits() -> None:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1500))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024))


def llama_template_probe(request_id: int, body: dict) -> dict:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{BACKEND_PORT}/apply-template",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        rendered = json.loads(response.read())
    prompt = rendered["prompt"]
    token_request = urllib.request.Request(
        f"http://127.0.0.1:{BACKEND_PORT}/tokenize",
        data=json.dumps({"content": prompt, "add_special": True, "parse_special": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(token_request, timeout=30) as response:
        tokenized = json.loads(response.read())
    return {
        "backend": "llama",
        "kind": "chat_template_probe",
        "request_id": request_id,
        "prompt": prompt,
        "prompt_token_ids": tokenized["tokens"],
        "token_count": len(tokenized["tokens"]),
    }


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    backend = os.environ["AB_BACKEND"]
    workspace = Path("/mnt/workspace")
    events: queue.Queue = queue.Queue()
    range_proc = None
    server_proc = None
    proxy = None
    proxy_thread = None
    ready_ms = None
    agent_exit = None
    cap_reason = None
    agent_stats: dict = {}
    captured_requests: list[dict] = []
    run_start = time.monotonic()
    emit("@@SUPERVISOR ", {
        "event": "trial_start",
        "backend": backend,
        "workspace": str(workspace),
        "limits": {
            "wall_seconds": MAX_WALL_SECONDS,
            "startup_seconds": MAX_STARTUP_SECONDS,
            "tool_seconds": MAX_TOOL_SECONDS,
            "tool_calls": MAX_TOOL_CALLS,
            "turns": MAX_TURNS,
            "http_exchanges": MAX_HTTP_EXCHANGES,
            "max_tokens_per_request": MAX_OUTPUT_TOKENS,
        },
    })
    try:
        if backend == "vbuf":
            range_command = [
                "/usr/bin/python3", "/tmp/runtime-tools/range_server.py",
                "--file", str(VBUF), "--host", "127.0.0.1", "--port", str(RANGE_PORT),
            ]
            range_proc = subprocess.Popen(
                range_command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            time.sleep(0.3)
            if range_proc.poll() is not None:
                raise RuntimeError(f"range server exited with status {range_proc.returncode}")
        command, server_env, _unused_range_command = backend_command(backend)
        server_proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=server_env, start_new_session=True,
        )
        threading.Thread(target=drain_process_output, args=(server_proc, backend, events), daemon=True).start()
        ready_seconds = wait_health(events, server_proc)
        ready_ms = round(ready_seconds * 1000, 1)
        if os.environ.get("AB_TEMPLATE_PROBE") == "1":
            with open("/tmp/runtime-tools/template-probe-request.json", encoding="utf-8") as request_file:
                probe_body = json.load(request_file)
            probe = llama_template_probe(0, probe_body)
            emit("@@SERVERLOG ", probe)
            agent_exit = 0
            agent_stats = {"template_probe_only": True, "prompt_token_count": probe["token_count"]}
            emit("@@SUPERVISOR ", {"event": "template_probe_complete", "backend": backend, "startup_ms": ready_ms})
        else:
            proxy = start_proxy(BACKEND_PORT, events)
            proxy_thread = threading.Thread(target=proxy.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            proxy_thread.start()
            with urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"local proxy health returned HTTP {response.status}")
            prepare_agent_config(workspace)
            emit("@@SUPERVISOR ", {"event": "backend_ready", "backend": backend, "startup_ms": ready_ms, "proxy_health": "PASS"})
            agent_exit, cap_reason, agent_stats, captured_requests = run_agent(workspace, events)
            if backend == "llama":
                for captured in captured_requests:
                    try:
                        probe = llama_template_probe(int(captured.get("id", 0)), captured["body"])
                        emit("@@SERVERLOG ", probe)
                    except Exception as error:
                        emit("@@SERVERLOG ", {
                            "backend": "llama", "kind": "chat_template_probe_error",
                            "request_id": captured.get("id"), "error": repr(error),
                        })
    except BaseException as error:
        emit("@@SUPERVISOR ", {"event": "trial_exception", "error": repr(error)})
        cap_reason = cap_reason or f"supervisor exception: {error!r}"
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        stop_process(server_proc)
        stop_process(range_proc)
        emit("@@SUPERVISOR ", {
            "event": "trial_end",
            "backend": backend,
            "agent_exit_code": agent_exit,
            "cap_reason": cap_reason,
            "ready_ms": ready_ms,
            "wall_seconds": round(time.monotonic() - run_start, 3),
            "agent_stats": agent_stats,
        })
    return 0 if agent_exit == 0 and cap_reason is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
