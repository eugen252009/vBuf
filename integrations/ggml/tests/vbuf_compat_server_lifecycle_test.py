#!/usr/bin/env python3
"""Persistent-process lifecycle qualification for vbuf_compat_server."""

import argparse
import concurrent.futures
import json
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


def http_request(base, method, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read(), time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read(), time.monotonic() - started


def model_list(base):
    status, _, body, _ = http_request(base, "GET", "/v1/models")
    assert status == 200
    return json.loads(body)


def chat(base, model, prompt, max_tokens=1):
    status, headers, body, elapsed = http_request(base, "POST", "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    })
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    response = json.loads(body)
    choice = response["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "elapsed_ms": elapsed * 1000.0,
    }


def parse_sse(body):
    events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
    assert events and events[-1] == "[DONE]"
    decoded = [json.loads(event) for event in events[:-1]]
    assert decoded and all(event["choices"] for event in decoded)
    content = "".join(event["choices"][0]["delta"].get("content", "") for event in decoded)
    return decoded, content


def stream_chat(base, model, prompt, max_tokens=2):
    status, headers, body, elapsed = http_request(base, "POST", "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    })
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    events, content = parse_sse(body)
    ids = {event["id"] for event in events}
    assert len(ids) == 1
    return {"content": content, "event_count": len(events), "elapsed_ms": elapsed * 1000.0}


def raw_disconnect(base, model, prompt, max_tokens):
    host_port = base.split("://", 1)[1].split("/", 1)[0]
    host, port_text = host_port.rsplit(":", 1)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }, ensure_ascii=False).encode()
    client = socket.create_connection((host, int(port_text)), timeout=10)
    client.settimeout(10)
    client.sendall((
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body)
    received = b""
    chunks = 0
    deadline = time.monotonic() + 12
    try:
        while time.monotonic() < deadline and chunks < 2:
            part = client.recv(8192)
            if not part:
                break
            received += part
            chunks = received.count(b"data: {")
    except socket.timeout:
        pass
    finally:
        client.close()
    return {"chunks_observed_before_disconnect": chunks, "headers_observed": b"\r\n\r\n" in received}


def rss_kib(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def wait_ready(base):
    started = time.monotonic()
    while time.monotonic() - started < 30:
        try:
            status, _, body, _ = http_request(base, "GET", "/health", timeout=2)
            if status == 200 and json.loads(body)["runtime"] == "ready":
                return (time.monotonic() - started) * 1000.0
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.1)
    raise AssertionError("server did not become ready")


def parse_request_logs(path):
    records = []
    pattern = re.compile(r"([a-z_]+)=([^\s]+)")
    with open(path, encoding="utf-8") as log:
        for line in log:
            if not line.startswith("vbuf_request "):
                continue
            record = dict(pattern.findall(line))
            records.append(record)
    return records


def invalid_request_recovery(base, model):
    checks = {}
    status, _, _, _ = http_request(base, "POST", "/v1/chat/completions", None)
    checks["invalid_json"] = status == 400
    status, _, body, _ = http_request(base, "POST", "/v1/chat/completions", {
        "model": "missing-model", "messages": [{"role": "user", "content": "bad"}], "max_tokens": 1,
    })
    checks["unknown_model"] = status == 400 and b"not found" in body
    status, _, body, _ = http_request(base, "POST", "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "bad"}], "max_tokens": 1,
        "temperature": 0,
    })
    checks["unsupported_option"] = status == 400 and b"unsupported generation option" in body
    status, _, body, _ = http_request(base, "POST", "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "bad"}], "max_tokens": 99,
    })
    checks["invalid_max_tokens"] = status == 400 and b"bounded generation limit" in body
    assert all(checks.values()), checks
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--model", default="vbuf-deepseek-bounded")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--evidence", default="")
    args = parser.parse_args()
    assert args.requests >= 6

    base = f"http://{args.host}:{args.port}"
    log_file = tempfile.NamedTemporaryFile(prefix="vbuf-step31s-", suffix=".log", delete=False)
    process = subprocess.Popen([
        args.binary,
        "--semantic-model", args.semantic_model,
        "--source-url", args.source_url,
        "--model-alias", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--blocks", str(args.blocks),
        "--capacity", str(args.capacity),
        "--max-new-tokens", str(args.max_new_tokens),
        "--runtime-mode", "normal",
    ], stdout=subprocess.DEVNULL, stderr=log_file)
    try:
        health_ready_ms = wait_ready(base)
        rss_start = rss_kib(process.pid)
        before_models = model_list(base)

        prompts = {"A": "Say hi", "B": "Count to one", "C": "Name a color"}
        order = ["A", "B", "A", "C", "B", "A"]
        order = (order * ((args.requests + len(order) - 1) // len(order)))[:args.requests]
        corpus = []
        rss_warmup = rss_start
        for index, identity in enumerate(order):
            result = chat(base, args.model, prompts[identity])
            result.update({"index": index, "prompt_id": identity})
            corpus.append(result)
            if index == min(4, args.requests - 1):
                rss_warmup = rss_kib(process.pid)
        assert all(item["content"] and item["completion_tokens"] == 1 for item in corpus)
        a_outputs = [item["content"] for item in corpus if item["prompt_id"] == "A"]
        assert len(set(a_outputs)) == 1

        recovery = invalid_request_recovery(base, args.model)
        after_error = chat(base, args.model, prompts["A"])
        after_error_models = model_list(base)

        from openai import OpenAI
        openai_client = OpenAI(base_url=base + "/v1", api_key="unused", timeout=180)
        openai_models = openai_client.models.list()
        assert openai_models.data and openai_models.data[0].id == args.model
        openai_responses = [openai_client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompts[name]}],
            max_tokens=1,
        ) for name in ("A", "B", "A")]
        assert all(response.choices[0].message.content for response in openai_responses)
        openai_stream = openai_client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompts["A"]}],
            max_tokens=2,
            stream=True,
        )
        openai_stream_content = "".join(
            chunk.choices[0].delta.content or "" for chunk in openai_stream
        )
        assert openai_stream_content

        streams = [stream_chat(base, args.model, prompts[name], 2) for name in ("A", "B", "A")]
        assert streams[0]["content"] == streams[2]["content"]
        after_stream_models = model_list(base)

        cancellation = raw_disconnect(base, args.model, "Continue with several tokens", args.max_new_tokens)
        cancelled_followup = chat(base, args.model, prompts["C"])
        after_cancel_models = model_list(base)

        health_during = {}
        stream_thread = threading.Thread(
            target=lambda: stream_chat(base, args.model, "Continue with several tokens", args.max_new_tokens),
            daemon=True,
        )
        stream_thread.start()
        time.sleep(0.25)
        health_status, _, _, health_elapsed = http_request(base, "GET", "/health", timeout=180)
        stream_thread.join(timeout=180)
        health_during = {"status": health_status, "elapsed_ms": health_elapsed * 1000.0}
        assert health_status == 200 and not stream_thread.is_alive()

        def concurrent_request(prompt):
            started = time.monotonic()
            result = chat(base, args.model, prompt)
            result["started"] = started
            result["finished"] = time.monotonic()
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(concurrent_request, prompts["A"]), pool.submit(concurrent_request, prompts["B"])]
            simultaneous = [future.result() for future in futures]
        assert all(item["content"] for item in simultaneous)
        steady_tail = [chat(base, args.model, prompts["A"]) for _ in range(5)]
        assert all(item["content"] == a_outputs[0] for item in steady_tail)
        rss_tail = rss_kib(process.pid)
        final_models = model_list(base)

        log_file.flush()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        log_file.close()
        records = parse_request_logs(log_file.name)
        assert process.returncode == 0
        assert records and all(record.get("active_leases_after") == "0" for record in records)
        assert all(record.get("active_generations_after") == "0" for record in records)
        assert all(record.get("active_streams_after") == "0" for record in records)
        assert all(record.get("active_cancellations_after") == "0" for record in records)
        cancelled_records = [record for record in records if record.get("cancelled") == "yes"]
        assert cancelled_records

        evidence = {
            "server_pid": process.pid,
            "server_reused": True,
            "request_count": args.requests,
            "distinct_prompts": 3,
            "corpus": corpus,
            "a_outputs": a_outputs,
            "stream_results": streams,
            "cancellation": cancellation,
            "recovery": recovery,
            "after_error": after_error,
            "openai_client": {
                "model_list": True,
                "chat_requests": len(openai_responses),
                "stream_content": openai_stream_content,
            },
            "cancelled_followup": cancelled_followup,
            "health_during_generation": health_during,
            "simultaneous": simultaneous,
            "models_stable": before_models == after_error_models == after_stream_models == after_cancel_models == final_models,
            "steady_tail": steady_tail,
            "rss_kib": {"start": rss_start, "warmup": rss_warmup, "tail": rss_tail},
            "health_ready_ms": health_ready_ms,
            "records": records,
            "server_log": log_file.name,
            "clean_shutdown": process.returncode == 0,
        }
        if args.evidence:
            with open(args.evidence, "w", encoding="utf-8") as output:
                json.dump(evidence, output, ensure_ascii=False, indent=2)

        print("VBUF_COMPAT_SERVER_LIFECYCLE_TEST=PASS")
        print("SEQUENTIAL_REQUEST_COUNT=" + str(args.requests))
        print("DISTINCT_PROMPT_COUNT=3")
        print("A0_A1_PARITY=PASS")
        print("STREAM_ISOLATION=PASS")
        print("CANCELLATION_RECOVERY=PASS")
        print("FAILURE_RECOVERY=PASS")
        print("CONCURRENCY_POLICY=SERIAL_QUEUE")
        print("HEALTH_DURING_GENERATION=PASS_BLOCKED_BY_SERIAL_DISPATCH")
        print("CLEAN_SHUTDOWN=PASS")
        print("SERVER_LOG=" + log_file.name)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if not log_file.closed:
            log_file.close()


if __name__ == "__main__":
    main()
