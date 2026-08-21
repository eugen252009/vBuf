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
    assert response["usage"]["completion_tokens"] == 1

    status, _, body = request(args.url, "POST", "/v1/completions", {
        "model": args.model, "prompt": "Say hi", "max_tokens": 1,
    })
    completion = json.loads(body)
    assert status == 200 and completion["object"] == "text_completion"

    status, _, body = request(args.url, "POST", "/v1/chat/completions", {
        **chat, "temperature": 0,
    })
    assert status == 400 and "unsupported generation option" in body.decode()

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
    assert all(json.loads(event)["choices"] for event in events[:-1])

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
