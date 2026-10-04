#!/usr/bin/env python3
"""Exact-artifact non-streaming native-Qwen tool roundtrip qualification."""

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
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_magic_number",
        "description": "Look up the fixed number for a supplied key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
    },
}
MESSAGES = [{
    "role": "user",
    "content": "You must call lookup_magic_number exactly once with key amber. Do not explain or answer in prose; call the function now.",
}]


def request(base, path, payload=None, method="POST"):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_ready(base, process, log_path):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited: " + log_path.read_text(errors="replace")[-5000:])
        try:
            code, body = request(base, "/health", method="GET")
            if code == 200 and json.loads(body).get("runtime") == "ready":
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError("server did not become ready")


def one_record(log_path, count):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        records = [line for line in log_path.read_text(errors="replace").splitlines()
                   if line.startswith("vbuf_request ")]
        if len(records) >= count:
            return records[-1]
        time.sleep(.05)
    raise TimeoutError("missing vbuf_request diagnostic")


def field(line, key):
    match = re.search(rf"(?:^| ){re.escape(key)}=([^ ]+)", line)
    if not match:
        raise AssertionError(f"missing {key}: {line}")
    return match.group(1)


def assert_isolated(record):
    for key in ("active_leases_after", "active_lease_bytes_after", "active_inflight_bytes_after",
                "active_generations_after"):
        assert field(record, key) == "0", (key, record)
    assert field(record, "runtime_creation_count") == "1"
    assert field(record, "qwen_model_upload_tensors") == "443"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(Path(args.payload).read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA, digest

    tool_request = {"model": MODEL, "messages": MESSAGES, "tools": [TOOL], "tool_choice": "auto", "max_tokens": 256}
    with tempfile.TemporaryDirectory(prefix="qwen-native-tool-") as temp:
        fixture = Path(temp) / "request.json"
        fixture.write_text(json.dumps(tool_request, separators=(",", ":")))
        direct = subprocess.run([
            args.direct, "--semantic-model", args.semantic, "--source-url", args.source_url,
            "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
            "--warmup", "0", "--requests", "3", "--native-tool-request", str(fixture),
        ], check=True, capture_output=True, text=True, timeout=1800)
        direct_lines = [line for line in direct.stdout.splitlines() if line.startswith("vbuf_direct_request")]
        assert len(direct_lines) == 3, direct.stdout
        direct_line = direct_lines[0]
        direct_hash = re.search(r"prompt_token_hash=([0-9a-f]+)", direct_line).group(1)
        direct_prompt_ids = re.search(r"prompt_token_ids=([^ ]*)", direct_line).group(1)
        native = bytes.fromhex(re.search(r"output=([0-9a-f]*)", direct_line).group(1)).decode()
        direct_token_ids = re.search(r"generated_token_ids=([^ ]*)", direct_line).group(1)
        direct_token_hash = re.search(r"generated_token_hash=([0-9a-f]+)", direct_line).group(1)
        assert direct_prompt_ids
        for direct_record in direct_lines:
            assert re.search(r"prompt_token_ids=([^ ]*)", direct_record).group(1) == direct_prompt_ids
            assert field(direct_record, "active_leases_after") == "0"
            assert field(direct_record, "active_inflight_bytes_after") == "0"
        for repeated in direct_lines[1:]:
            assert re.search(r"prompt_token_hash=([0-9a-f]+)", repeated).group(1) == direct_hash
            assert re.search(r"generated_token_hash=([0-9a-f]+)", repeated).group(1) == direct_token_hash
            assert re.search(r"generated_token_ids=([^ ]*)", repeated).group(1) == direct_token_ids
            assert re.search(r"output=([0-9a-f]*)", repeated).group(1) == re.search(r"output=([0-9a-f]*)", direct_line).group(1)
        assert "<tool_call>" in native, native
        parsed_native = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", native, re.S)
        assert parsed_native, native
        direct_call = json.loads(parsed_native.group(1))
        assert direct_call["name"] == TOOL["function"]["name"]
        assert isinstance(direct_call["arguments"], dict)

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        base = f"http://127.0.0.1:{port}"
        log = Path(temp) / "server.log"
        with log.open("w") as output:
            env = os.environ.copy()
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env["VBUF_QUALIFICATION_DUMP_TOKEN_IDS"] = "1"
            proc = subprocess.Popen([
                args.server, "--semantic-model", args.semantic, "--source-url", args.source_url,
                "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
                "--host", "127.0.0.1", "--port", str(port),
            ], stdout=subprocess.DEVNULL, stderr=output, env=env)
            try:
                wait_ready(base, proc, log)
                http_results = []
                for index in range(3):
                    status, body = request(base, "/v1/chat/completions", tool_request)
                    first = json.loads(body)
                    assert status == 200, first
                    http_results.append(first)
                    one_record(log, index + 1)
                first = http_results[0]
                choice = first["choices"][0]
                assert choice["finish_reason"] == "tool_calls", first
                message = choice["message"]
                assert message["content"] is None
                assert len(message["tool_calls"]) == 1
                call = message["tool_calls"][0]
                assert call["type"] == "function" and call["function"]["name"] == direct_call["name"]
                arguments = json.loads(call["function"]["arguments"])
                assert isinstance(arguments, dict) and arguments == direct_call["arguments"]
                assert call["id"]
                records = [line for line in log.read_text(errors="replace").splitlines()
                           if line.startswith("vbuf_request ")]
                assert len(records) == 3
                record = records[0]
                assert field(record, "prompt_token_ids") == direct_prompt_ids
                assert field(record, "generated_token_ids") == direct_token_ids
                assert int(field(record, "prompt_token_hash"), 16) == int(direct_hash, 16), (
                    f"direct prompt hash {direct_hash} != HTTP {field(record, 'prompt_token_hash')} "
                    f"(HTTP record: {record})")
                assert int(field(record, "generated_token_hash"), 16) == int(direct_token_hash, 16)
                for repeated, repeated_record in zip(http_results[1:], records[1:]):
                    repeated_call = repeated["choices"][0]["message"]["tool_calls"][0]
                    assert repeated["choices"][0]["finish_reason"] == "tool_calls"
                    assert repeated_call["function"]["name"] == call["function"]["name"]
                    assert json.loads(repeated_call["function"]["arguments"]) == arguments
                    assert field(repeated_record, "prompt_token_ids") == direct_prompt_ids
                    assert field(repeated_record, "generated_token_ids") == direct_token_ids
                    assert int(field(repeated_record, "prompt_token_hash"), 16) == int(direct_hash, 16)
                    assert int(field(repeated_record, "generated_token_hash"), 16) == int(direct_token_hash, 16)

                history = MESSAGES + [message, {
                    "role": "tool", "tool_call_id": call["id"], "content": "42",
                }]
                continuation = {"model": MODEL, "messages": history, "tools": [TOOL], "tool_choice": "none", "max_tokens": 256}
                status, body = request(base, "/v1/chat/completions", continuation)
                final = json.loads(body)
                assert status == 200, final
                final_choice = final["choices"][0]
                assert final_choice["finish_reason"] in ("stop", "length")
                assert isinstance(final_choice["message"]["content"], str) and final_choice["message"]["content"]
                assert not final_choice["message"].get("tool_calls")

                none_request = {
                    "model": MODEL, "messages": [{"role": "user", "content": "Reply briefly with OK."}],
                    "tools": [TOOL], "tool_choice": "none", "max_tokens": 32,
                }
                none_status, none_body = request(base, "/v1/chat/completions", none_request)
                none_response = json.loads(none_body)
                assert none_status == 200 and none_response["choices"][0]["message"]["content"]
                assert not none_response["choices"][0]["message"].get("tool_calls")

                malformed = [
                    {**tool_request, "tools": [{"type": "not-function", "function": TOOL["function"]}]},
                    {**tool_request, "tools": [TOOL, TOOL]},
                    {**tool_request, "tool_choice": {"type": "function", "function": {"name": TOOL["function"]["name"]}}},
                    {"model": MODEL, "messages": MESSAGES + [{"role": "tool", "tool_call_id": "orphan", "content": "x"}]},
                    {"model": MODEL, "messages": MESSAGES + [
                        {"role": "assistant", "content": None, "tool_calls": [{
                            "id": "call_expected", "type": "function", "function": {
                                "name": TOOL["function"]["name"], "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "call_wrong", "content": "x"}]},
                    {**tool_request, "tools": [{"type": "function", "function": {
                        "name": "large_description", "description": "word " * 1800,
                        "parameters": {"type": "object"}}}]},
                ]
                for invalid in malformed:
                    code, invalid_body = request(base, "/v1/chat/completions", invalid)
                    assert code == 400, (code, invalid_body[:500])
                assert b"executable context capacity" in invalid_body
                stream_status, stream_body = request(base, "/v1/chat/completions", {**tool_request, "stream": True})
                assert stream_status == 200 and b'"tool_calls"' in stream_body
                assert b'"finish_reason":"tool_calls"' in stream_body and b"data: [DONE]" in stream_body
                good_status, good_body = request(base, "/v1/chat/completions", {
                    "model": MODEL, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 1,
                })
                assert good_status == 200 and json.loads(good_body)["choices"][0]["message"]["content"] is not None

                barrier = threading.Barrier(2)
                def overlap(payload):
                    barrier.wait()
                    return request(base, "/v1/chat/completions", payload)
                text_request = {"model": MODEL, "messages": [{"role": "user", "content": "Reply briefly with OK."}], "max_tokens": 1}
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    overlap_results = list(pool.map(overlap, [tool_request, text_request]))
                assert all(status == 200 for status, _ in overlap_results)
                overlap_ids = {json.loads(body)["id"] for _, body in overlap_results}
                admission_deadline = time.monotonic() + 30
                overlap_records = []
                terminal_lines = []
                admission_log = ""
                while time.monotonic() < admission_deadline:
                    admission_log = log.read_text(errors="replace")
                    overlap_records = [line for line in admission_log.splitlines()
                                       if line.startswith("vbuf_request ") and
                                       any(f"id={request_id} " in line for request_id in overlap_ids)]
                    terminal_lines = [line for line in admission_log.splitlines()
                                      if line.startswith("vbuf_admission event=terminal ") and
                                      any(f"id={request_id} " in line for request_id in overlap_ids)]
                    if len(overlap_records) == 2 and len(terminal_lines) == 2:
                        break
                    time.sleep(.05)
                assert len(overlap_records) == 2 and len(terminal_lines) == 2, admission_log
                runtime_ranges = [(int(re.search(r"runtime_start_ns=(\d+)", line).group(1)),
                                   int(re.search(r"runtime_end_ns=(\d+)", line).group(1)))
                                  for line in overlap_records]
                assert runtime_ranges[0][1] <= runtime_ranges[1][0] or runtime_ranges[1][1] <= runtime_ranges[0][0]
                terminal_waits = [int(re.search(r"admission_wait_ns=(\d+)", line).group(1))
                                  for line in terminal_lines]
                assert max(terminal_waits) > 0
                all_request_records = [line for line in log.read_text(errors="replace").splitlines()
                                       if line.startswith("vbuf_request ")]
                for request_record in all_request_records:
                    assert_isolated(request_record)
                print(json.dumps({
                    "roundtrip": "PASS", "artifact_sha256": digest,
                    "tool": call["function"]["name"], "arguments": arguments,
                    "tool_call_id": call["id"], "finish_reason": choice["finish_reason"],
                    "synthetic_tool_result": "42", "final_content": final_choice["message"]["content"],
                    "final_finish_reason": final_choice["finish_reason"],
                    "prompt_token_hash_direct_http": direct_hash,
                    "prompt_token_ids_direct_http": direct_prompt_ids,
                    "generated_token_hash_direct_http": field(record, "generated_token_hash"),
                    "direct_generated_token_ids": direct_token_ids,
                    "direct_native_output": native,
                }, ensure_ascii=False))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def hashlib_token_ids(csv):
    # Match the existing FNV-style 64-bit token_hash implemented in the server.
    value = 1469598103934665603
    tokens = [] if not csv else [int(item) for item in csv.split(",")]
    for token in tokens:
        value ^= token
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


if __name__ == "__main__":
    main()
