#!/usr/bin/env python3
"""HTTP contract checks for the bounded vBuf compatibility server."""

import argparse
import json
import socket
import time
import urllib.error
import urllib.request


def request(base, method, path, payload=None):
    url = base.rstrip("/") + path
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="vbuf-deepseek-bounded")
    args = parser.parse_args()

    status, _, body = request(args.url, "GET", "/health")
    assert status == 200 and json.loads(body)["runtime"] == "ready"

    status, _, body = request(args.url, "GET", "/v1/models")
    models = json.loads(body)
    assert status == 200 and models["data"][0]["id"] == args.model
    assert args.model in body.decode() and "/tmp/" not in body.decode()

    chat = {"model": args.model, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 1}
    status, headers, body = request(args.url, "POST", "/v1/chat/completions", chat)
    response = json.loads(body)
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"]
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 1

    empty_tools = {**chat, "tools": [], "tool_choice": "none"}
    status, _, body = request(args.url, "POST", "/v1/chat/completions", empty_tools)
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"]

    status, _, body = request(args.url, "POST", "/v1/completions", {
        "model": args.model, "prompt": "Say hi", "max_tokens": 1,
    })
    completion = json.loads(body)
    assert status == 200 and completion["object"] == "text_completion"
    assert completion["choices"][0]["finish_reason"] == "length"

    status, headers, body = request(args.url, "POST", "/v1/completions", {
        "model": args.model, "prompt": "Say hi", "max_tokens": 1, "stream": True,
    })
    completion_stream = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    completion_events = [line[6:] for line in completion_stream.splitlines() if line.startswith("data: ")]
    assert completion_events[-1] == "[DONE]"
    completion_chunks = [json.loads(event) for event in completion_events[:-1]]
    assert all(chunk["object"] == "text_completion" for chunk in completion_chunks)
    assert completion_chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert "".join(chunk["choices"][0]["text"] for chunk in completion_chunks) == completion["choices"][0]["text"]

    status, _, body = request(args.url, "POST", "/v1/chat/completions", {
        **chat, "temperature": 0,
    })
    assert status == 400 and "unsupported generation option" in body.decode()

    tool_request = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Use add"}],
        "tools": [{"type": "function", "function": {
            "name": "add", "parameters": {"type": "object"},
        }}],
        "tool_choice": "auto",
        "max_tokens": 1,
    }
    status, _, body = request(args.url, "POST", "/v1/chat/completions", tool_request)
    assert status == 400 and b"unsupported_feature" in body and b"no qualified native tool-call adapter" in body

    invalid_choice = {**tool_request, "tool_choice": {
        "type": "function", "function": {"name": "missing"},
    }}
    status, _, body = request(args.url, "POST", "/v1/chat/completions", invalid_choice)
    assert status == 400 and b"unavailable function" in body

    malformed_tool_call = {
        "model": args.model,
        "messages": [{"role": "user", "content": "x"}, {"role": "assistant",
            "content": None, "tool_calls": [{"id": "call_x", "type": "function",
                "function": {"name": "add", "arguments": "{"}}]}],
    }
    status, _, body = request(args.url, "POST", "/v1/chat/completions", malformed_tool_call)
    assert status == 400 and b"invalid_request_error" in body

    invalid_tool_history = {
        "model": args.model,
        "messages": [{"role": "tool", "tool_call_id": "orphan", "content": "x"}],
        "max_tokens": 1,
    }
    status, _, body = request(args.url, "POST", "/v1/chat/completions", invalid_tool_history)
    assert status == 400 and b"invalid_conversation_state" in body

    status, _, body = request(args.url, "POST", "/v1/completions", {
        "model": args.model, "prompt": "hello", "max_tokens": 1,
        "tools": tool_request["tools"],
    })
    assert status == 400 and b"supported only by /v1/chat/completions" in body

    host_port = args.url.split("://", 1)[1].split("/", 1)[0]
    host, port_text = host_port.rsplit(":", 1)
    oversized = socket.create_connection((host, int(port_text)), timeout=10)
    oversized.sendall((
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}\r\nContent-Length: {4 * 1024 * 1024 + 1}\r\n"
        "Connection: close\r\n\r\n"
    ).encode())
    oversized_response = bytearray()
    while True:
        chunk = oversized.recv(4096)
        if not chunk:
            break
        oversized_response.extend(chunk)
    oversized.close()
    assert oversized_response.startswith(b"HTTP/1.1 400 ")
    assert b"HTTP request body too large" in oversized_response

    status, _, body = request(args.url, "POST", "/v1/chat/completions", {
        **chat, "model": "unknown-model",
    })
    assert status == 400 and "not found" in body.decode()

    status, _, body = request(args.url, "POST", "/v1/chat/completions", None)
    assert status == 400

    invalid = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=b"{",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(invalid, timeout=10)
        raise AssertionError("invalid JSON unexpectedly succeeded")
    except urllib.error.HTTPError as error:
        assert error.code == 400

    invalid_escape = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=(f'{{"model":"{args.model}","messages":[{{"role":"user","content":"\\uZZZZ"}}]}}').encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(invalid_escape, timeout=10)
        raise AssertionError("invalid JSON escape unexpectedly succeeded")
    except urllib.error.HTTPError as error:
        assert error.code == 400

    status, headers, body = request(args.url, "POST", "/v1/chat/completions", {
        **chat, "stream": True,
    })
    stream = body.decode()
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    events = [line[6:] for line in stream.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    parsed_events = [json.loads(event) for event in events[:-1]]
    assert all(event["choices"] for event in parsed_events)
    streamed_text = "".join(event["choices"][0]["delta"].get("content", "") or "" for event in parsed_events)
    assert streamed_text == response["choices"][0]["message"]["content"]
    assert parsed_events[-1]["choices"][0]["finish_reason"] == response["choices"][0]["finish_reason"]

    first_status, _, first_body = request(args.url, "POST", "/v1/chat/completions", chat)
    second_status, _, second_body = request(args.url, "POST", "/v1/chat/completions", chat)
    assert first_status == second_status == 200
    assert json.loads(first_body)["choices"][0]["message"]["content"] == json.loads(second_body)["choices"][0]["message"]["content"]

    host_port = args.url.split("://", 1)[1].split("/", 1)[0]
    host, port_text = host_port.rsplit(":", 1)
    client = socket.create_connection((host, int(port_text)), timeout=10)
    cancellation_body = json.dumps({**chat, "max_tokens": 4, "stream": True}).encode()
    client.sendall((
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}\r\nContent-Type: application/json\r\nContent-Length: {len(cancellation_body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + cancellation_body)
    client.recv(512)
    client.close()
    for _ in range(30):
        try:
            status, _, _ = request(args.url, "GET", "/health")
            if status == 200:
                break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    else:
        raise AssertionError("server did not recover after client cancellation")

    print("VBUF_COMPAT_SERVER_HTTP_TEST=PASS")


if __name__ == "__main__":
    main()
