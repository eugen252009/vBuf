#!/usr/bin/env python3
"""Exact-artifact text-only qualification for the canonical Qwen HTTP path."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED_SHA = "f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31"
MODEL = "qwen3-14b-q4km"


def http(base, method, path, payload=None, raw=None, timeout=180):
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode())
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def direct_tokens(executable, semantic, source, completion):
    command = [
        executable, "--semantic-model", semantic, "--source-url", source,
        "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "2",
        "--warmup", "0", "--requests", "1", "--prompt", "Say hi",
    ]
    if completion:
        command.append("--completion")
    completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=600)
    line = next(line for line in completed.stdout.splitlines() if line.startswith("vbuf_direct_request"))
    return {
        "prompt_hash": re.search(r"prompt_token_hash=([0-9a-f]+)", line).group(1),
        "token_hash": re.search(r"generated_token_hash=([0-9a-f]+)", line).group(1),
        "text": bytes.fromhex(re.search(r"output=([0-9a-f]*)", line).group(1)).decode("utf-8"),
    }


def wait_server(base, process, log_path):
    end = time.monotonic() + 300
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("Qwen server exited during startup:\n" + log_path.read_text(errors="replace")[-6000:])
        try:
            status, _, body = http(base, "GET", "/health", timeout=2)
            if status == 200 and json.loads(body).get("runtime") == "ready":
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError("Qwen server did not become ready:\n" + log_path.read_text(errors="replace")[-6000:])


def records(log_path, request_id=None):
    lines = log_path.read_text(errors="replace").splitlines()
    result = [line for line in lines if line.startswith("vbuf_request ")]
    if request_id is not None:
        result = [line for line in result if f"id={request_id} " in line]
    return result


def wait_records(log_path, minimum):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = records(log_path)
        if len(result) >= minimum:
            return result
        time.sleep(0.01)
    raise TimeoutError(f"server emitted only {len(records(log_path))} request records; expected {minimum}")


def field(line, name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise AssertionError(f"missing {name} in server diagnostic: {line}")
    return match.group(1)


def chat_request(messages, max_tokens=2, **extra):
    return {"model": MODEL, "messages": messages, "max_tokens": max_tokens, **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--nonstream-requests", type=int, default=20)
    parser.add_argument("--stream-requests", type=int, default=10)
    args = parser.parse_args()

    hasher = hashlib.sha256()
    with open(args.payload, "rb") as artifact:
        for block in iter(lambda: artifact.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    assert digest == EXPECTED_SHA, f"artifact SHA mismatch: {digest}"

    # Run direct controls before starting the server, so the 12 GB GPU never
    # holds two Qwen model runtimes simultaneously.
    direct_chat = direct_tokens(args.direct, args.semantic, args.source_url, False)
    direct_completion = direct_tokens(args.direct, args.semantic, args.source_url, True)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    base = f"http://127.0.0.1:{port}"
    stdout = tempfile.NamedTemporaryFile(prefix="qwen-http-out-", delete=False)
    stderr = tempfile.NamedTemporaryFile(prefix="qwen-http-err-", delete=False)
    log_path = Path(stderr.name)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    process = subprocess.Popen([
        args.server, "--semantic-model", args.semantic, "--source-url", args.source_url,
        "--model-alias", MODEL, "--max-new-tokens", "8", "--host", "127.0.0.1",
        "--port", str(port),
    ], stdout=stdout, stderr=stderr, env=env)
    metrics = {"nonstream": 0, "stream": 0, "failure_recovery": False, "concurrent": False}
    try:
        wait_server(base, process, log_path)
        status, _, body = http(base, "GET", "/v1/models")
        models = json.loads(body)
        assert status == 200 and models == {"object": "list", "data": [
            {"id": MODEL, "object": "model", "owned_by": "vbuf"}]}

        first_chat = chat_request([{"role": "user", "content": "Say hi"}], 2)
        status, headers, body = http(base, "POST", "/v1/chat/completions", first_chat)
        chat = json.loads(body)
        assert status == 200 and headers.get_content_type() == "application/json"
        assert chat["object"] == "chat.completion" and chat["choices"][0]["index"] == 0
        assert chat["choices"][0]["message"]["role"] == "assistant"
        assert chat["choices"][0]["finish_reason"] == "length"
        direct_line = wait_records(log_path, 1)[-1]
        assert field(direct_line, "prompt_token_hash") == direct_chat["prompt_hash"]
        assert field(direct_line, "generated_token_hash") == direct_chat["token_hash"]
        assert chat["choices"][0]["message"]["content"] == direct_chat["text"]
        assert chat["usage"]["completion_tokens"] == 2

        completion_request = {"model": MODEL, "prompt": "Say hi", "max_tokens": 2}
        status, _, body = http(base, "POST", "/v1/completions", completion_request)
        completion = json.loads(body)
        assert status == 200 and completion["object"] == "text_completion"
        completion_line = wait_records(log_path, 2)[-1]
        assert field(completion_line, "prompt_token_hash") == direct_completion["prompt_hash"]
        assert field(completion_line, "generated_token_hash") == direct_completion["token_hash"]
        assert completion["choices"][0]["text"] == direct_completion["text"]

        # Explicit empty tools are text-only; non-empty tools remain rejected.
        empty_tools_before = len(records(log_path))
        status, _, body = http(base, "POST", "/v1/chat/completions", {
            **first_chat, "tools": [], "tool_choice": "none"})
        assert status == 200 and json.loads(body)["choices"][0]["message"]["role"] == "assistant"
        empty_tools_line = wait_records(log_path, empty_tools_before + 1)[-1]
        assert field(empty_tools_line, "prompt_token_hash") == direct_chat["prompt_hash"]
        assert field(empty_tools_line, "generated_token_hash") == direct_chat["token_hash"]
        status, _, body = http(base, "POST", "/v1/chat/completions", {
            **first_chat, "tools": [{"type": "function", "function": {
                "name": "x", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "x"}}})
        assert status == 400 and b"unsupported_feature" in body
        assert b"cannot enforce a named tool_choice" in body

        # Both SSE endpoint forms use valid OpenAI frames and reconstruct text.
        status, headers, body = http(base, "POST", "/v1/chat/completions", {
            **first_chat, "max_tokens": 1, "stream": True})
        stream_text = body.decode("utf-8")
        assert status == 200 and headers.get_content_type() == "text/event-stream"
        chat_events = [line[6:] for line in stream_text.splitlines() if line.startswith("data: ")]
        assert chat_events[-1] == "[DONE]"
        chat_frames = [json.loads(event) for event in chat_events[:-1]]
        assert all(frame["object"] == "chat.completion.chunk" for frame in chat_frames)
        reconstructed = "".join(frame["choices"][0]["delta"].get("content", "") or "" for frame in chat_frames)
        status, _, body = http(base, "POST", "/v1/chat/completions", {
            **first_chat, "max_tokens": 1})
        assert status == 200 and reconstructed == json.loads(body)["choices"][0]["message"]["content"]
        assert chat_frames[-1]["choices"][0]["finish_reason"] == "length"

        status, headers, body = http(base, "POST", "/v1/completions", {
            **completion_request, "max_tokens": 1, "stream": True})
        completion_events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
        assert status == 200 and headers.get_content_type() == "text/event-stream"
        assert completion_events[-1] == "[DONE]"
        completion_frames = [json.loads(event) for event in completion_events[:-1]]
        assert all(frame["object"] == "text_completion" for frame in completion_frames)
        completion_reconstructed = "".join(frame["choices"][0]["text"] for frame in completion_frames)
        status, _, body = http(base, "POST", "/v1/completions", {
            "model": MODEL, "prompt": "Say hi", "max_tokens": 1})
        assert status == 200 and completion_reconstructed == json.loads(body)["choices"][0]["text"]
        metrics["stream"] += 1

        # Stateless multi-turn prompt: explicit history changes prompt tokens;
        # a subsequent request without it gets a fresh prompt/session.
        history = [
            {"role": "user", "content": "My code word is ORANGE. Reply OK."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "What was my code word?"},
        ]
        before_history = len(records(log_path))
        status, _, body = http(base, "POST", "/v1/chat/completions", chat_request(history, 8))
        assert status == 200 and json.loads(body)["choices"][0]["message"]["content"]
        history_hash = field(wait_records(log_path, before_history + 1)[-1], "prompt_token_hash")
        before_no_history = len(records(log_path))
        status, _, body = http(base, "POST", "/v1/chat/completions", chat_request([
            {"role": "user", "content": "What was my code word?"}], 8))
        assert status == 200
        no_history_hash = field(wait_records(log_path, before_no_history + 1)[-1], "prompt_token_hash")
        assert history_hash != no_history_hash

        # Failures are client errors and do not poison the retained runtime.
        invalids = [
            ("POST", "/v1/chat/completions", None),
            ("POST", "/v1/chat/completions", {**first_chat, "model": "wrong"}),
            ("POST", "/v1/chat/completions", {"model": MODEL, "messages": []}),
            ("POST", "/v1/chat/completions", {"model": MODEL, "messages": [
                {"role": "tool", "content": "x"}]}),
            ("POST", "/v1/chat/completions", {**first_chat, "max_tokens": 0}),
            ("POST", "/v1/chat/completions", {**first_chat, "temperature": 0}),
            ("POST", "/v1/completions", {**completion_request, "top_p": 0.9}),
        ]
        for method, path, payload in invalids:
            if payload is None:
                status, _, _ = http(base, method, path, raw=b"{")
            else:
                status, _, _ = http(base, method, path, payload)
            assert status == 400, (path, payload, status)
        overflow = {"model": MODEL, "prompt": "word " * 1200, "max_tokens": 8}
        status, _, body = http(base, "POST", "/v1/completions", overflow)
        assert status == 400 and b"executable context capacity" in body
        status, _, body = http(base, "POST", "/v1/completions", {
            **completion_request, "n": 2})
        assert status == 400 and b"unsupported completion request field" in body
        status, _, body = http(base, "POST", "/v1/chat/completions", first_chat)
        assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == chat["choices"][0]["message"]["content"]
        metrics["failure_recovery"] = True

        # A/B/A/B/A isolated sessions with deterministic one-token requests.
        a = chat_request([{"role": "user", "content": "Reply with one letter A."}], 1)
        b = chat_request([{"role": "user", "content": "Reply with one letter B."}], 1)
        sequence = []
        for payload in (a, b, a, b, a):
            status, _, body = http(base, "POST", "/v1/chat/completions", payload)
            assert status == 200
            sequence.append(json.loads(body)["choices"][0]["message"]["content"])
            metrics["nonstream"] += 1
        assert sequence[0] == sequence[2] == sequence[4]
        assert sequence[1] == sequence[3]

        # Minimum requested stability series; sessions are created/destroyed
        # independently while one model runtime remains resident.
        for index in range(args.nonstream_requests):
            payload = chat_request([{"role": "user", "content": f"Say item {index % 2}."}], 1)
            status, _, body = http(base, "POST", "/v1/chat/completions", payload)
            assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] is not None
            metrics["nonstream"] += 1
        for index in range(args.stream_requests):
            payload = chat_request([{"role": "user", "content": f"Stream item {index % 2}."}], 1, stream=True)
            status, headers, body = http(base, "POST", "/v1/chat/completions", payload)
            events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
            assert status == 200 and headers.get_content_type() == "text/event-stream"
            assert events[-1] == "[DONE]"
            metrics["stream"] += 1

        # Send two HTTP inference requests from synchronized threads. The
        # server's FIFO inference gate admits only one canonical call at once.
        barrier = threading.Barrier(2)
        def overlap(text):
            barrier.wait()
            return http(base, "POST", "/v1/chat/completions", chat_request([
                {"role": "user", "content": text}], 8))[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(overlap, ["Concurrent request A", "Concurrent request B"]))
        assert statuses == [200, 200]
        admission_deadline = time.monotonic() + 15
        admission_log = ""
        while time.monotonic() < admission_deadline:
            admission_log = log_path.read_text(errors="replace")
            if "queued_generations=2" in admission_log and re.search(
                r"event=terminal .*admission_wait_ns=[1-9][0-9]*", admission_log):
                break
            time.sleep(0.01)
        assert "queued_generations=2" in admission_log
        waits = [int(value) for value in re.findall(
            r"event=terminal .*admission_wait_ns=([0-9]+)", admission_log)]
        assert waits and max(waits) > 100_000_000
        metrics["concurrent"] = True

        # Drop an active SSE client, require cancellation cleanup, then prove
        # that the retained runtime accepts another request.
        before_disconnect = len(records(log_path))
        disconnect = socket.create_connection(("127.0.0.1", port), timeout=30)
        disconnect.settimeout(90)
        request_bytes = json.dumps(chat_request([
            {"role": "user", "content": "Generate a longer answer for disconnect testing."}], 8, stream=True)).encode()
        disconnect.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nConnection: close\r\nContent-Length: " +
            str(len(request_bytes)).encode() + b"\r\n\r\n" + request_bytes)
        received = b""
        while b"\r\n\r\n" not in received:
            received += disconnect.recv(4096)
        while b"data: " not in received:
            received += disconnect.recv(4096)
        disconnect.close()
        deadline = time.monotonic() + 120
        disconnect_lines = []
        while time.monotonic() < deadline:
            disconnect_lines = records(log_path)[before_disconnect:]
            if any("cancelled=yes" in line for line in disconnect_lines):
                break
            time.sleep(0.25)
        assert any("cancelled=yes" in line for line in disconnect_lines), disconnect_lines
        status, _, body = http(base, "GET", "/health")
        assert status == 200 and json.loads(body)["runtime"] == "ready"
        status, _, body = http(base, "POST", "/v1/chat/completions", first_chat)
        assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == chat["choices"][0]["message"]["content"]

        all_lines = records(log_path)
        assert len(all_lines) >= 1 + 1 + 2 + 2 + 5 + args.nonstream_requests + args.stream_requests + 2
        qwen_lines = [line for line in all_lines if "qwen_model_upload_tensors=443" in line]
        assert len(qwen_lines) == len(all_lines)
        assert all(field(line, "qwen_model_upload_bytes") == "8995793920" for line in qwen_lines)
        assert all(field(line, "runtime_creation_count") == "1" for line in qwen_lines)
        assert all(field(line, "active_generations_after") == "0" for line in qwen_lines)
        free_values = [int(field(line, "post_run_free_vram_bytes")) for line in qwen_lines]
        assert free_values and max(free_values) - min(free_values) < 512 * 1024 * 1024
        assert len(set(free_values[-min(10, len(free_values)):])) <= 3
        startup = log_path.read_text(errors="replace")
        assert "model_runtime_creations=1 qwen_model_upload_tensors=443 qwen_model_upload_bytes=8995793920" in startup
        print(json.dumps({
            "QWEN3_OPENAI_HTTP_QUALIFICATION": "PASS",
            "artifact_sha256": digest,
            "chat_direct_token_hash": direct_chat["token_hash"],
            "completion_direct_token_hash": direct_completion["token_hash"],
            "nonstream_requests": metrics["nonstream"],
            "stream_requests": metrics["stream"],
            "client_disconnect_cancellation": True,
            "fresh_sessions": len(all_lines),
            "model_runtime_creation_count": 1,
            "model_upload_tensors": 443,
            "model_upload_bytes": 8995793920,
            "post_run_free_vram_min_bytes": min(free_values),
            "post_run_free_vram_max_bytes": max(free_values),
            "peak_vram_min_bytes": min(int(field(line, "peak_vram_bytes")) for line in qwen_lines),
            "peak_vram_max_bytes": max(int(field(line, "peak_vram_bytes")) for line in qwen_lines),
            "overlapping_requests": "SERIALIZED_BY_FIFO_GATE",
        }, sort_keys=True))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stdout.close()
        stderr.close()


if __name__ == "__main__":
    main()
