#!/usr/bin/env python3
"""Exact-artifact live qualification for native Qwen tool-call SSE."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

EXPECTED_SHA = "f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31"
MODEL = "qwen3-14b-q4km"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_magic_number",
        "description": "Look up the fixed number for a supplied key.",
        "parameters": {
            "type": "object", "properties": {"key": {"type": "string"}},
            "required": ["key"], "additionalProperties": False,
        },
    },
}
OTHER_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_calendar_entry",
        "description": "Look up a calendar entry by label.",
        "parameters": {
            "type": "object", "properties": {"label": {"type": "string"}},
            "required": ["label"], "additionalProperties": False,
        },
    },
}
TOOLS = [TOOL, OTHER_TOOL]
MESSAGES = [{
    "role": "user",
    "content": "You must call lookup_magic_number exactly once with key amber. Do not explain or answer in prose; call the function now.",
}]
TOOL_REQUEST = {"model": MODEL, "messages": MESSAGES, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 256}


def perform_http_request(base, path, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(base + path, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def wait_ready(base, process, log_path):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server exited: " + log_path.read_text(errors="replace")[-5000:])
        try:
            status, _, body = perform_http_request(base, "/health")
            if status == 200 and json.loads(body).get("runtime") == "ready":
                return
        except Exception:
            time.sleep(.5)
    raise TimeoutError("server did not become ready")


def log_lines(log_path, prefix):
    return [line for line in log_path.read_text(errors="replace").splitlines() if line.startswith(prefix)]


def wait_log_count(log_path, prefix, count, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = log_lines(log_path, prefix)
        if len(lines) >= count:
            return lines
        time.sleep(.05)
    raise TimeoutError(f"timed out waiting for {count} {prefix!r} records: {log_path.read_text(errors='replace')[-6000:]}")


def field(line, name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise AssertionError(f"missing {name}: {line}")
    return match.group(1)


def parse_sse(body):
    records = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data:"):
            records.append(line[5:].lstrip())
    assert records and records[-1] == "[DONE]", records
    return records


def reconstruct_text_stream(records):
    pieces = []
    finishes = []
    for record in records[:-1]:
        event = json.loads(record)
        assert event["object"] == "chat.completion.chunk"
        assert len(event["choices"]) == 1 and event["choices"][0]["index"] == 0
        choice = event["choices"][0]
        delta = choice["delta"]
        assert "tool_calls" not in delta
        if "content" in delta:
            pieces.append(delta["content"] or "")
        if choice["finish_reason"] is not None:
            finishes.append(choice["finish_reason"])
    assert any(pieces) and len(finishes) == 1 and finishes[0] in ("stop", "length")
    return "".join(pieces), finishes[0]


def reconstruct_tool_stream(records):
    assert records[-1] == "[DONE]"
    calls = {}
    role_count = 0
    finishes = []
    response_ids = set()
    content = ""
    for record in records[:-1]:
        event = json.loads(record)
        response_ids.add(event["id"])
        assert event["object"] == "chat.completion.chunk"
        assert len(event["choices"]) == 1 and event["choices"][0]["index"] == 0
        choice = event["choices"][0]
        delta = choice["delta"]
        if delta.get("role") is not None:
            assert delta["role"] == "assistant"
            role_count += 1
        content += delta.get("content") or ""
        if choice["finish_reason"] is not None:
            finishes.append(choice["finish_reason"])
        for fragment in delta.get("tool_calls", []):
            index = fragment["index"]
            call = calls.setdefault(index, {"id": "", "type": None, "name": "", "arguments": "",
                                             "id_parts": 0, "type_parts": 0})
            if "id" in fragment:
                call["id"] += fragment["id"]
                call["id_parts"] += 1
            if "type" in fragment:
                call["type"] = fragment["type"]
                call["type_parts"] += 1
            function = fragment.get("function", {})
            call["name"] += function.get("name", "")
            call["arguments"] += function.get("arguments", "")
    assert len(response_ids) == 1 and role_count == 1
    assert finishes == ["tool_calls"]
    assert sorted(calls) == [0]
    call = calls[0]
    assert call["id"] and call["id_parts"] == 1
    assert call["type"] == "function" and call["type_parts"] == 1
    assert call["name"] == TOOL["function"]["name"]
    arguments = json.loads(call["arguments"])
    assert isinstance(arguments, dict)
    assert not content
    return call, finishes[0], response_ids.pop()


def raw_post(port, payload, extra_headers=(), close_reset=False, wait_for=None,
             wait_for_count=None, log_path=None):
    body = json.dumps(payload, separators=(",", ":")).encode()
    sock = socket.create_connection(("127.0.0.1", port), timeout=30)
    sock.settimeout(900)
    headers = [
        "POST /v1/chat/completions HTTP/1.1", "Host: 127.0.0.1", "Connection: close",
        "Content-Type: application/json", f"Content-Length: {len(body)}", *extra_headers, "", "",
    ]
    sock.sendall("\r\n".join(headers).encode() + body)
    if wait_for is not None:
        assert log_path is not None
        deadline = time.monotonic() + 30
        matched = None
        while time.monotonic() < deadline:
            waiting = [line for line in log_lines(log_path, "vbuf_admission ") if wait_for in line]
            if len(waiting) >= (wait_for_count or 1):
                matched = waiting[-1]
                break
            time.sleep(.02)
        assert matched, log_path.read_text(errors="replace")[-6000:]
        time.sleep(.1)
    if close_reset:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()
        return b""
    received = bytearray()
    while True:
        try:
            part = sock.recv(65536)
        except ConnectionResetError:
            break
        if not part:
            break
        received.extend(part)
    sock.close()
    return bytes(received)


def wait_request_record(log_path, request_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for line in log_lines(log_path, "vbuf_request "):
            if field(line, "id") == request_id:
                return line
        time.sleep(.05)
    raise TimeoutError(f"missing request diagnostic for {request_id}: {log_path.read_text(errors='replace')[-6000:]}")


def assert_clean_record(line):
    for name in ("active_leases_after", "active_lease_bytes_after", "active_inflight_bytes_after",
                 "active_generations_after", "active_streams_after", "active_cancellations_after"):
        assert field(line, name) == "0", (name, line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(Path(args.payload).read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA, digest

    with tempfile.TemporaryDirectory(prefix="qwen-native-tool-sse-") as temp:
        temp = Path(temp)
        fixture = temp / "request.json"
        fixture.write_text(json.dumps(TOOL_REQUEST, separators=(",", ":")))
        direct = subprocess.run([
            args.direct, "--semantic-model", args.semantic, "--source-url", args.source_url,
            "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
            "--warmup", "0", "--requests", "1", "--native-tool-request", str(fixture),
        ], check=True, capture_output=True, text=True, timeout=1800)
        direct_line = next(line for line in direct.stdout.splitlines() if line.startswith("vbuf_direct_request "))
        direct_output = bytes.fromhex(field(direct_line, "output")).decode()
        native_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", direct_output, re.S)
        assert native_match, direct_output
        direct_call = json.loads(native_match.group(1))
        direct_prompt_ids = field(direct_line, "prompt_token_ids")
        direct_generated_ids = field(direct_line, "generated_token_ids")
        assert len(direct_prompt_ids.split(",")) + 256 <= 1032

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.close()
        base = f"http://127.0.0.1:{port}"
        log_path = temp / "server.log"
        with log_path.open("w") as output:
            env = os.environ.copy()
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env["VBUF_QUALIFICATION_DUMP_TOKEN_IDS"] = "1"
            process = subprocess.Popen([
                args.server, "--semantic-model", args.semantic, "--source-url", args.source_url,
                "--model-alias", MODEL, "--blocks", "40", "--max-new-tokens", "256",
                "--host", "127.0.0.1", "--port", str(port), "--enable-qualification-faults",
            ], stdout=subprocess.DEVNULL, stderr=output, env=env)
            try:
                wait_ready(base, process, log_path)

                def http_request(base, path, payload=None, headers=None):
                    status, response_headers, body = perform_http_request(base, path, payload, headers)
                    if status == 200 and path in ("/v1/chat/completions", "/v1/completions"):
                        if response_headers.get_content_type() == "text/event-stream":
                            response_id = json.loads(parse_sse(body)[0])["id"]
                        else:
                            response_id = json.loads(body)["id"]
                        wait_request_record(log_path, response_id)
                    return status, response_headers, body

                # Baseline text SSE with omitted and explicitly empty tools.
                plain_text = {"model": MODEL, "messages": [{"role": "user", "content": "Reply briefly with OK."}],
                              "max_tokens": 1, "stream": True}
                status, headers, body = http_request(base, "/v1/chat/completions", plain_text)
                assert status == 200 and headers.get_content_type() == "text/event-stream"
                plain_records = parse_sse(body)
                plain_text_out, plain_finish = reconstruct_text_stream(plain_records)
                empty_text = {**plain_text, "tools": []}
                status, headers, body = http_request(base, "/v1/chat/completions", empty_text)
                assert status == 200 and headers.get_content_type() == "text/event-stream"
                empty_records = parse_sse(body)
                empty_text_out, empty_finish = reconstruct_text_stream(empty_records)
                assert empty_text_out == plain_text_out and empty_finish == plain_finish

                # The native result is parsed completely before any SSE data is sent.
                before_tool = len(log_lines(log_path, "vbuf_request "))
                status, headers, body = http_request(base, "/v1/chat/completions", {**TOOL_REQUEST, "stream": True})
                assert status == 200 and headers.get_content_type() == "text/event-stream"
                stream_records = parse_sse(body)
                streamed_call, streamed_finish, stream_id = reconstruct_tool_stream(stream_records)
                assert streamed_finish == "tool_calls" and json.loads(streamed_call["arguments"]) == direct_call["arguments"]
                assert streamed_call["name"] == direct_call["name"] == TOOL["function"]["name"]
                stream_log = wait_log_count(log_path, "vbuf_request ", before_tool + 1)[before_tool]
                assert field(stream_log, "prompt_token_ids") == direct_prompt_ids, (
                    field(stream_log, "prompt_token_ids"), direct_prompt_ids)
                assert field(stream_log, "generated_token_ids") == direct_generated_ids, (
                    field(stream_log, "generated_token_ids"), direct_generated_ids)

                # Exact non-stream control; request-local call IDs intentionally differ.
                status, headers, body = http_request(base, "/v1/chat/completions", TOOL_REQUEST)
                assert status == 200
                nonstream = json.loads(body)["choices"][0]
                nonstream_call = nonstream["message"]["tool_calls"][0]
                assert nonstream["finish_reason"] == streamed_finish == "tool_calls"
                assert nonstream_call["type"] == streamed_call["type"] == "function"
                assert nonstream_call["function"]["name"] == streamed_call["name"]
                assert nonstream_call["function"]["arguments"] == streamed_call["arguments"]
                assert nonstream_call["id"] != streamed_call["id"]
                nonstream_log = wait_log_count(log_path, "vbuf_request ", before_tool + 2)[before_tool + 1]
                assert field(nonstream_log, "prompt_token_ids") == direct_prompt_ids
                assert field(nonstream_log, "generated_token_ids") == direct_generated_ids

                # The reconstructed streamed call is accepted by the qualified role=tool continuation path.
                assistant_history = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": streamed_call["id"], "type": "function",
                                    "function": {"name": streamed_call["name"],
                                                 "arguments": streamed_call["arguments"]}}],
                }
                continuation = {"model": MODEL,
                    "messages": MESSAGES + [assistant_history,
                        {"role": "tool", "tool_call_id": streamed_call["id"], "content": "42"}],
                    "tools": TOOLS, "tool_choice": "none", "max_tokens": 256}
                status, _, body = http_request(base, "/v1/chat/completions", continuation)
                final = json.loads(body)["choices"][0]
                assert status == 200 and final["finish_reason"] in ("stop", "length")
                assert final["message"]["content"].strip() == "42" and not final["message"].get("tool_calls")

                # Sequence isolation: ordinary text, non-stream tool, text SSE, repeated tool SSE.
                status, _, body = http_request(base, "/v1/chat/completions", {
                    "model": MODEL, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 1,
                })
                assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] is not None
                status, headers, body = http_request(base, "/v1/chat/completions", plain_text)
                assert status == 200 and headers.get_content_type() == "text/event-stream"
                assert "tool_calls" not in "".join(parse_sse(body)[:-1])
                status, headers, body = http_request(base, "/v1/chat/completions", {**TOOL_REQUEST, "stream": True})
                repeated_call, _, repeated_response_id = reconstruct_tool_stream(parse_sse(body))
                assert status == 200 and repeated_call["name"] == streamed_call["name"]
                assert repeated_call["arguments"] == streamed_call["arguments"]
                assert repeated_call["id"] != streamed_call["id"] and repeated_response_id != stream_id

                # Context overflow is rejected before headers/data or model inference.
                oversized_tool = {"type": "function", "function": {
                    "name": "large_description", "description": "word " * 1800,
                    "parameters": {"type": "object"}}}
                before_capacity = len(log_lines(log_path, "vbuf_request "))
                status, headers, body = http_request(base, "/v1/chat/completions", {
                    **TOOL_REQUEST, "tools": [oversized_tool], "stream": True,
                })
                assert status == 400 and headers.get_content_type() == "application/json"
                assert b"executable context capacity" in body
                assert len(log_lines(log_path, "vbuf_request ")) == before_capacity

                # Invalid streamed tool options fail before SSE; a valid text request still works.
                status, headers, body = http_request(base, "/v1/chat/completions", {
                    **TOOL_REQUEST, "stream": True,
                    "tool_choice": {"type": "function", "function": {"name": TOOL["function"]["name"]}},
                })
                assert status == 400 and headers.get_content_type() == "application/json"
                assert b"cannot enforce a named tool_choice" in body
                status, _, body = http_request(base, "/v1/chat/completions", plain_text)
                assert status == 200 and b"[DONE]" in body

                # Real client disconnect before a complete native call: no SSE headers/data are committed.
                before_disconnect = len(log_lines(log_path, "vbuf_request "))
                waiting_count = len(log_lines(log_path, "vbuf_admission event=waiting "))
                raw_post(port, {**TOOL_REQUEST, "stream": True}, close_reset=True,
                         wait_for="event=waiting", wait_for_count=waiting_count + 1, log_path=log_path)
                disconnect_record = wait_log_count(log_path, "vbuf_request ", before_disconnect + 1)[before_disconnect]
                assert "cancelled=yes" in disconnect_record and "finish_reason=cancelled" in disconnect_record
                assert_clean_record(disconnect_record)
                assert len(log_lines(log_path, "vbuf_admission event=waiting ")) > waiting_count
                status, _, body = http_request(base, "/v1/chat/completions", plain_text)
                assert status == 200 and b"[DONE]" in body

                # A deterministic writer fault after two SSE records models send failure after partial output.
                before_write_failure = len(log_lines(log_path, "vbuf_request "))
                raw = raw_post(port, {**TOOL_REQUEST, "stream": True},
                    extra_headers=("X-VBuf-Qualification-SSE-Write-Fail-After: 2",))
                assert raw.startswith(b"HTTP/1.1 200 OK") and b"text/event-stream" in raw.split(b"\r\n\r\n", 1)[0]
                assert b'"tool_calls"' in raw and b'"arguments"' not in raw
                assert b'"finish_reason":"tool_calls"' not in raw and b"data: [DONE]" not in raw
                failed_write_record = wait_log_count(log_path, "vbuf_request ", before_write_failure + 1)[before_write_failure]
                assert "cancelled=yes" in failed_write_record, failed_write_record
                assert_clean_record(failed_write_record)
                status, _, body = http_request(base, "/v1/chat/completions", plain_text)
                assert status == 200 and b"[DONE]" in body

                # FIFO overlap: streamed tool + text, then streamed tool + non-stream tool.
                def pair(first, second):
                    barrier = threading.Barrier(2)
                    def run(payload):
                        barrier.wait()
                        return http_request(base, "/v1/chat/completions", payload)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        return list(pool.map(run, [first, second]))

                before_pair = len(log_lines(log_path, "vbuf_request "))
                pair_one = pair({**TOOL_REQUEST, "stream": True}, plain_text)
                assert pair_one[0][0] == pair_one[1][0] == 200
                parse_sse(pair_one[0][2])
                wait_log_count(log_path, "vbuf_request ", before_pair + 2)
                first_pair_records = log_lines(log_path, "vbuf_request ")[before_pair:before_pair + 2]
                before_pair_two = before_pair + 2
                pair_two = pair({**TOOL_REQUEST, "stream": True}, TOOL_REQUEST)
                assert pair_two[0][0] == pair_two[1][0] == 200
                parse_sse(pair_two[0][2])
                wait_log_count(log_path, "vbuf_request ", before_pair_two + 2)
                pair_records = log_lines(log_path, "vbuf_request ")[before_pair_two:before_pair_two + 2]
                for records in (first_pair_records, pair_records):
                    ranges = [(int(field(line, "runtime_start_ns")), int(field(line, "runtime_end_ns"))) for line in records]
                    assert ranges[0][1] <= ranges[1][0] or ranges[1][1] <= ranges[0][0]
                    ids = [field(line, "id") for line in records]
                    admission_lines = [line for line in log_lines(log_path, "vbuf_admission event=terminal ")
                                       if field(line, "id") in ids]
                    assert len(admission_lines) == 2 and max(int(field(line, "admission_wait_ns"))
                                                            for line in admission_lines) > 0

                all_records = log_lines(log_path, "vbuf_request ")
                for line in all_records:
                    assert_clean_record(line)
                print(json.dumps({
                    "stream_tool_qualification": "PASS", "artifact_sha256": digest,
                    "stream_frames": len(stream_records), "tool_index": 0,
                    "tool_call_id": streamed_call["id"], "function": streamed_call["name"],
                    "arguments": streamed_call["arguments"], "arguments_fragments": 1,
                    "finish_reason": streamed_finish, "done": True,
                    "direct_prompt_ids_match": True, "direct_generated_ids_match": True,
                    "nonstream_semantic_match": True, "tool_result": final["message"]["content"],
                    "disconnect_cancelled": True, "writer_fault_after_records": 2,
                    "capacity_rejected_before_inference": True,
                    "fifo_pairs_serialized": 2, "request_records": len(all_records),
                }, ensure_ascii=False))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
