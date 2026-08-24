#!/usr/bin/env python3
"""Step 31V SERIAL_QUEUE timing and lifecycle qualification."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


FIELDS = re.compile(r"([a-z_]+)=([^\s]+)")


def parse_log(text):
    records = []
    controls = []
    connections = []
    for line in text.splitlines():
        if line.startswith("vbuf_request "):
            records.append(dict(FIELDS.findall(line)))
        elif line.startswith("vbuf_control_request "):
            controls.append(dict(FIELDS.findall(line)))
        elif line.startswith("vbuf_connection "):
            connections.append(dict(FIELDS.findall(line)))
    return records, controls, connections


def http_request(base, payload=None, path="/v1/chat/completions", timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        method="GET" if data is None else "POST",
        headers={} if data is None else {"Content-Type": "application/json"},
    )
    started = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, body, started, time.monotonic_ns()
    except urllib.error.HTTPError as error:
        return error.code, error.read(), started, time.monotonic_ns()


def wait_ready(base):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            status, body, _, _ = http_request(base, path="/health", timeout=2)
            if status == 200 and json.loads(body)["runtime"] == "ready":
                return
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise AssertionError("server did not become ready")


def payload(model, max_tokens, stream=False, failure_after=None):
    value = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": max_tokens,
    }
    if stream:
        value["stream"] = True
    if failure_after is not None:
        value["vbuf_source_failure_after_requests"] = failure_after
    return value


def response_id(body):
    match = re.search(rb'"id":"(chatcmpl-vbuf-[0-9]+)"', body)
    return match.group(1).decode() if match else ""


def chat(base, model, max_tokens, stream=False, failure_after=None):
    status, body, started, finished = http_request(
        base, payload(model, max_tokens, stream, failure_after))
    return {
        "status": status,
        "id": response_id(body),
        "started_ns": started,
        "finished_ns": finished,
        "body": body.decode(errors="replace"),
    }


def start_server(args, port, max_tokens=8, capacity=None, faults=False):
    log = tempfile.NamedTemporaryFile(prefix="vbuf-step31v-", suffix=".log", mode="w+")
    command = [
        args.binary,
        "--semantic-model", args.semantic_model,
        "--source-url", args.source_url,
        "--model-alias", args.model,
        "--host", args.host,
        "--port", str(port),
        "--blocks", str(args.blocks),
        "--capacity", str(args.capacity if capacity is None else capacity),
        "--max-new-tokens", str(max_tokens),
        "--runtime-mode", "normal",
    ]
    if faults:
        command.append("--enable-qualification-faults")
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": args.ld_library_path},
    )
    base = f"http://{args.host}:{port}"
    wait_ready(base)
    return process, log, base


def stop_server(process, log):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    log.flush()
    log.seek(0)
    text = log.read()
    log.close()
    assert process.returncode == 0, f"server exited with {process.returncode}"
    return parse_log(text)


def timeline(client, record):
    accepted = int(record["accepted_ns"])
    runtime_start = int(record["runtime_start_ns"])
    runtime_end = int(record["runtime_end_ns"])
    first_token = int(record["first_token_ns"])
    return {
        "queue_ms": (accepted - client["started_ns"]) / 1_000_000.0,
        "dispatch_to_runtime_ms": (runtime_start - accepted) / 1_000_000.0,
        "ttft_ms": None if first_token == 0 else (first_token - client["started_ns"]) / 1_000_000.0,
        "service_ms": (runtime_end - runtime_start) / 1_000_000.0,
        "client_total_ms": (client["finished_ns"] - client["started_ns"]) / 1_000_000.0,
        "request_index": int(record["request_index"]),
        "completed_layers": int(record["completed_layers"]),
        "completed_positions": int(record["completed_positions"]),
    }


def run_pair(args, port, name, a_tokens, b_tokens, stream=False, warmup=False):
    process, log, base = start_server(args, port, max_tokens=max(a_tokens, b_tokens))
    try:
        warmup_result = None
        if warmup:
            warmup_result = chat(base, args.model, 1)
            assert warmup_result["status"] == 200
        with ThreadPoolExecutor(max_workers=2) as pool:
            a_future = pool.submit(chat, base, args.model, a_tokens, stream)
            time.sleep(0.02)
            b_future = pool.submit(chat, base, args.model, b_tokens, stream)
            a = a_future.result()
            b = b_future.result()
    finally:
        records, controls, connections = stop_server(process, log)
    assert a["status"] == b["status"] == 200, (name, a, b)
    by_id = {record.get("id"): record for record in records}
    assert a["id"] in by_id and b["id"] in by_id, (name, records)
    assert int(by_id[a["id"]]["request_index"]) < int(by_id[b["id"]]["request_index"])
    return {
        "name": name,
        "stream": stream,
        "warmup": warmup_result,
        "a": {**a, "timeline": timeline(a, by_id[a["id"]])},
        "b": {**b, "timeline": timeline(b, by_id[b["id"]])},
        "makespan_ms": (max(a["finished_ns"], b["finished_ns"]) - min(a["started_ns"], b["started_ns"])) / 1_000_000.0,
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def read_http_response(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        part = sock.recv(8192)
        if not part:
            return data
        data += part
    header_end = data.index(b"\r\n\r\n") + 4
    headers = data[:header_end].lower()
    if b"transfer-encoding: chunked" in headers:
        while b"\r\n0\r\n\r\n" not in data[header_end:]:
            part = sock.recv(8192)
            if not part:
                break
            data += part
    else:
        length_match = re.search(rb"content-length:\s*(\d+)", headers)
        length = int(length_match.group(1)) if length_match else 0
        while len(data) - header_end < length:
            part = sock.recv(8192)
            if not part:
                break
            data += part
    return data


def send_raw(sock, model, max_tokens, stream=False, failure_after=None):
    body = json.dumps(payload(model, max_tokens, stream, failure_after)).encode()
    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {sock.getpeername()[0]}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body
    started = time.monotonic_ns()
    sock.sendall(request)
    return started


def run_fairness(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(chat, base, args.model, 8)
            time.sleep(0.1)
            followers = []
            for _ in range(4):
                follower = socket.create_connection((args.host, port), timeout=30)
                follower.settimeout(900)
                started = send_raw(follower, args.model, 1)
                followers.append((follower, started))
            active_result = active.result()
            follower_results = []
            for follower, started in followers:
                body = read_http_response(follower)
                follower_results.append({"id": response_id(body), "started_ns": started, "status": int(body.split(b" ", 2)[1])})
                follower.close()
    finally:
        records, controls, connections = stop_server(process, log)
    assert active_result["status"] == 200
    assert all(item["status"] == 200 for item in follower_results)
    by_id = {record.get("id"): record for record in records}
    execution_order = [record.get("id") for record in sorted(records, key=lambda item: int(item["request_index"]))]
    expected = [active_result["id"]] + [item["id"] for item in follower_results]
    assert execution_order == expected, (execution_order, expected)
    return {
        "arrival_order": expected,
        "execution_order": execution_order,
        "queue_order": "FIFO",
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def run_control_plane(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.1)
            health_future = pool.submit(http_request, base, path="/health")
            models_future = pool.submit(http_request, base, path="/v1/models")
            active = active_future.result()
            health = health_future.result()
            models = models_future.result()
    finally:
        records, controls, connections = stop_server(process, log)
    assert active["status"] == health[0] == models[0] == 200
    return {
        "active": active,
        "health": {"status": health[0], "started_ns": health[2], "finished_ns": health[3]},
        "models": {"status": models[0], "started_ns": models[2], "finished_ns": models[3]},
        "blocked_health": health[3] >= active["finished_ns"],
        "blocked_models": models[3] >= active["finished_ns"],
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def run_queued_cancel(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.1)
            queued = socket.create_connection((args.host, port), timeout=30)
            queued.settimeout(30)
            send_raw(queued, args.model, 1)
            queued.close()
            active = active_future.result()
            recovery = pool.submit(chat, base, args.model, 1).result()
    finally:
        records, controls, connections = stop_server(process, log)
    assert active["status"] == recovery["status"] == 200
    by_id = {record.get("id"): record for record in records}
    assert recovery["id"] in by_id
    assert len(records) in (2, 3)
    assert int(by_id[recovery["id"]]["request_index"]) == len(records)
    return {
        "active_status": active["status"],
        "recovery_status": recovery["status"],
        "queued_entered_runtime": len(records) == 3,
        "queued_cancellation_gap": len(records) == 3,
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def run_active_cancel(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    active_socket = None
    follower = None
    try:
        active_socket = socket.create_connection((args.host, port), timeout=30)
        active_socket.settimeout(900)
        send_raw(active_socket, args.model, 8, stream=True)
        time.sleep(0.1)
        follower = socket.create_connection((args.host, port), timeout=30)
        follower.settimeout(900)
        send_raw(follower, args.model, 1)
        active_socket.close()
        active_socket = None
        follower_body = read_http_response(follower)
        follower.close()
        follower = None
    finally:
        if active_socket is not None:
            active_socket.close()
        if follower is not None:
            follower.close()
        records, controls, connections = stop_server(process, log)
    follower_id = response_id(follower_body)
    by_id = {record.get("id"): record for record in records}
    assert follower_id in by_id and by_id[follower_id]["cancelled"] == "no"
    cancelled = [record for record in records if record["cancelled"] == "yes"]
    assert cancelled
    return {
        "follower_status": int(follower_body.split(b" ", 2)[1]),
        "active_cancelled": True,
        "follower_after_cancel": True,
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def run_fault_follower(args, port):
    process, log, base = start_server(args, port, max_tokens=1, faults=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fault = pool.submit(chat, base, args.model, 1, False, 1)
            time.sleep(0.05)
            follower = pool.submit(chat, base, args.model, 1)
            fault_result = fault.result()
            follower_result = follower.result()
    finally:
        records, controls, connections = stop_server(process, log)
    assert fault_result["status"] == 500 and follower_result["status"] == 200
    follower_record = next(record for record in records if record["id"] == follower_result["id"])
    fault_record = min(records, key=lambda record: int(record["request_index"]))
    assert int(fault_record["request_index"]) < int(follower_record["request_index"])
    assert fault_record["source_failure_injected"] == "yes"
    return {
        "fault_status": fault_result["status"],
        "follower_status": follower_result["status"],
        "follower_after_fault": True,
        "records": records,
        "controls": controls,
        "connections": connections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--model", default="vbuf-step31v")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18140)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    results = {
        "short_short": run_pair(args, args.port, "short_short", 1, 1, warmup=True),
        "long_short": run_pair(args, args.port + 1, "long_short", 8, 1, warmup=True),
        "long_long": run_pair(args, args.port + 2, "long_long", 8, 8, warmup=True),
        "short_short_stream": run_pair(args, args.port + 3, "short_short_stream", 2, 2, True, True),
        "control_plane": run_control_plane(args, args.port + 4),
        "fairness": run_fairness(args, args.port + 5),
        "queued_cancel": run_queued_cancel(args, args.port + 6),
        "active_cancel": run_active_cancel(args, args.port + 7),
        "fault_follower": run_fault_follower(args, args.port + 8),
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(results, output, ensure_ascii=False, indent=2)
    print("VBUF_STEP31V_SERIAL_QUEUE_QUALIFICATION=PASS")
    for name in ("short_short", "long_short", "long_long"):
        item = results[name]
        print(f"{name.upper()}_A_QUEUE_MS={item['a']['timeline']['queue_ms']:.3f}")
        print(f"{name.upper()}_B_QUEUE_MS={item['b']['timeline']['queue_ms']:.3f}")
        print(f"{name.upper()}_MAKESPAN_MS={item['makespan_ms']:.3f}")
    print("HEALTH_BLOCKED=" + str(results["control_plane"]["blocked_health"]).upper())
    print("MODELS_BLOCKED=" + str(results["control_plane"]["blocked_models"]).upper())
    print("QUEUE_ORDER=" + results["fairness"]["queue_order"])
    print("QUEUED_CANCEL=PASS")
    print("ACTIVE_CANCEL_FOLLOWER=PASS")
    print("FAULT_FOLLOWER=PASS")
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
