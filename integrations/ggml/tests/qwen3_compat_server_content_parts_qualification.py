#!/usr/bin/env python3
"""Exact-artifact qualification for OpenAI text message content-part arrays."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

EXPECTED_SHA = "f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31"
MODEL = "Qwen_Qwen3-14B-Q4_K_M"
PROMPT = "Reply with exactly: PI_VBUF_OK"


def http(base, path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        base + path, data=data, method="GET" if data is None else "POST",
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def field(line, name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise AssertionError(f"missing {name} in diagnostic: {line}")
    return match.group(1)


def direct_control(executable, semantic, source):
    result = subprocess.run([
        executable, "--semantic-model", semantic, "--source-url", source,
        "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
        "--warmup", "0", "--requests", "1", "--prompt", PROMPT,
    ], check=True, capture_output=True, text=True, timeout=1800)
    return next(line for line in result.stdout.splitlines() if line.startswith("vbuf_direct_request "))


def parse_sse(body):
    records = [line[5:].lstrip() for line in body.decode("utf-8").splitlines() if line.startswith("data:")]
    assert records and records[-1] == "[DONE]", records[-4:]
    return records


def reconstruct_text(records):
    pieces = []
    finishes = []
    for record in records[:-1]:
        frame = json.loads(record)
        assert frame["object"] == "chat.completion.chunk"
        choice = frame["choices"][0]
        assert choice["index"] == 0 and "tool_calls" not in choice["delta"]
        pieces.append(choice["delta"].get("content") or "")
        if choice["finish_reason"] is not None:
            finishes.append(choice["finish_reason"])
    assert len(finishes) == 1 and finishes[0] in ("stop", "length")
    return "".join(pieces), finishes[0], json.loads(records[0])["id"]


def wait_server(base, process, log_path):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited during startup: " + log_path.read_text(errors="replace")[-6000:])
        try:
            status, _, body = http(base, "/health", timeout=3)
            if status == 200 and json.loads(body).get("runtime") == "ready":
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not become ready")


def wait_record(log_path, request_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        lines = [line for line in log_path.read_text(errors="replace").splitlines()
                 if line.startswith("vbuf_request ") and field(line, "id") == request_id]
        if lines:
            return lines[0]
        time.sleep(0.05)
    raise TimeoutError(f"missing inference record {request_id}: {log_path.read_text(errors='replace')[-6000:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(Path(args.payload).read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA, f"unexpected payload SHA-256: {digest}"
    direct = direct_control(args.direct, args.semantic, args.source_url)
    direct_prompt_ids = field(direct, "prompt_token_ids")
    direct_prompt_serialized = field(direct, "prompt_serialized_hex")
    direct_generated_ids = field(direct, "generated_token_ids")
    direct_text = bytes.fromhex(field(direct, "output")).decode("utf-8")
    assert "PI_VBUF_OK" in direct_text, repr(direct_text)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="qwen-content-parts-") as temp:
        log_path = Path(temp) / "server.log"
        with log_path.open("w") as log:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["VBUF_QUALIFICATION_DUMP_TOKEN_IDS"] = "1"
            process = subprocess.Popen([
                args.server, "--semantic-model", args.semantic, "--source-url", args.source_url,
                "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
                "--host", "127.0.0.1", "--port", str(port),
            ], stdout=subprocess.DEVNULL, stderr=log, env=env)
            try:
                wait_server(base, process, log_path)
                status, _, body = http(base, "/v1/models")
                assert status == 200 and json.loads(body)["data"][0]["id"] == MODEL

                def check_inference(payload, stream=False):
                    status, headers, body = http(base, "/v1/chat/completions", payload)
                    assert status == 200, (status, body)
                    if stream:
                        assert headers.get_content_type() == "text/event-stream"
                        records = parse_sse(body)
                        text, finish, request_id = reconstruct_text(records)
                        response = {"text": text, "finish": finish}
                    else:
                        assert headers.get_content_type() == "application/json"
                        decoded = json.loads(body)
                        response = {"text": decoded["choices"][0]["message"]["content"],
                                    "finish": decoded["choices"][0]["finish_reason"],
                                    "prompt_tokens": decoded["usage"]["prompt_tokens"],
                                    "completion_tokens": decoded["usage"]["completion_tokens"]}
                        request_id = decoded["id"]
                    return response, wait_record(log_path, request_id)

                string_payload = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256}
                parts_payload = {"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT}]}], "max_tokens": 256}
                string_response, string_log = check_inference(string_payload)
                parts_response, parts_log = check_inference(parts_payload)
                for line in (string_log, parts_log):
                    assert field(line, "prompt_token_ids") == direct_prompt_ids
                    assert field(line, "prompt_serialized_hex") == direct_prompt_serialized
                    assert field(line, "generated_token_ids") == direct_generated_ids
                assert string_response["text"] == parts_response["text"] == direct_text
                assert string_response["finish"] == parts_response["finish"]
                assert string_response["prompt_tokens"] == parts_response["prompt_tokens"]
                assert string_response["completion_tokens"] == parts_response["completion_tokens"]

                # Multiple adjacent text parts concatenate with no inserted separator.
                multi = {"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]}], "max_tokens": 4}
                joined = {"model": MODEL, "messages": [{"role": "user", "content": "foobar"}], "max_tokens": 4}
                multi_response, multi_log = check_inference(multi)
                joined_response, joined_log = check_inference(joined)
                for name in ("prompt_token_ids", "prompt_serialized_hex", "generated_token_ids"):
                    assert field(multi_log, name) == field(joined_log, name), name
                assert multi_response == joined_response

                # OpenAI text SSE has the same reconstructed result and model inputs.
                for use_parts in (False, True):
                    payload = parts_payload if use_parts else string_payload
                    response, line = check_inference({**payload, "stream": True}, stream=True)
                    assert response["text"] == direct_text
                    assert field(line, "prompt_token_ids") == direct_prompt_ids
                    assert field(line, "prompt_serialized_hex") == direct_prompt_serialized
                    assert field(line, "generated_token_ids") == direct_generated_ids

                # A near-limit content array tokenizes normalized text, not JSON syntax.
                near_text = "word " * 1000
                near = {"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": near_text}]}], "max_tokens": 8}
                near_string = {"model": MODEL, "messages": [{"role": "user", "content": near_text}], "max_tokens": 8}
                near_response, near_log = check_inference(near)
                near_string_response, near_string_log = check_inference(near_string)
                near_prompt_tokens = int(field(near_log, "prompt_tokens"))
                assert 980 <= near_prompt_tokens and near_prompt_tokens + 8 <= 1032, near_prompt_tokens
                assert near_response == near_string_response
                for name in ("prompt_token_ids", "prompt_serialized_hex", "generated_token_ids"):
                    assert field(near_log, name) == field(near_string_log, name), name
                assert near_response["prompt_tokens"] == near_prompt_tokens

                before_bad = len([line for line in log_path.read_text(errors="replace").splitlines()
                                  if line.startswith("vbuf_request ")])
                bad_contents = [
                    ([{"type": "image_url", "image_url": {"url": "https://invalid.example/image.png"}}], b"unsupported message content part type"),
                    ([{"type": "unknown_future_type"}], b"unsupported message content part type"),
                    ([{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "x"}}], b"unsupported message content part type"),
                ]
                for content, expected in bad_contents:
                    status, _, error = http(base, "/v1/chat/completions", {
                        "model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 1})
                    assert status == 400 and expected in error, (status, error)
                after_bad = len([line for line in log_path.read_text(errors="replace").splitlines()
                                 if line.startswith("vbuf_request ")])
                assert before_bad == after_bad, (before_bad, after_bad)
                recovery, recovery_log = check_inference(string_payload)
                assert recovery["text"] == direct_text
                assert field(recovery_log, "prompt_token_ids") == direct_prompt_ids

                overflow = {"model": MODEL, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "word " * 1200}]}], "max_tokens": 8}
                status, _, error = http(base, "/v1/chat/completions", overflow)
                assert status == 400 and b"executable context capacity" in error, (status, error)
                recovery, recovery_log = check_inference(string_payload)
                assert recovery["text"] == direct_text
                assert field(recovery_log, "prompt_token_ids") == direct_prompt_ids

                print("QWEN_CONTENT_PARTS_QUALIFICATION=PASS")
                print(f"artifact_sha256={digest}")
                print(f"string_and_single_text_part_prompt_tokens={len(direct_prompt_ids.split(','))}")
                print(f"serialized_prompt_hex={direct_prompt_serialized}")
                print(f"generated_token_ids={direct_generated_ids}")
                print(f"generated_text={direct_text!r}")
                print(f"multiple_text_parts=foobar prompt_ids_match=yes generated_ids_match=yes")
                print(f"near_capacity_prompt_tokens={near_prompt_tokens} plus_generation=8 capacity=1032")
                print("text_sse_string_and_parts=match unsupported_parts=fail_closed recovery=pass")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
