#!/usr/bin/env python3
"""Step 31U matched overhead and source-failure qualification."""

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


FIELD_PATTERN = re.compile(r"([a-z_]+)=([^\s]+)")


def parse_records(text):
    records = []
    for line in text.splitlines():
        if line.startswith("vbuf_direct_request ") or line.startswith("vbuf_request "):
            records.append(dict(FIELD_PATTERN.findall(line)))
    return records


def request(base, payload=None, path="/v1/chat/completions", timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    method = "GET" if data is None else "POST"
    headers = {} if data is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_ready(base):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            status, body = request(base, path="/health", timeout=2)
            if status == 200 and body.get("runtime") == "ready":
                return
        except (OSError, TimeoutError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise AssertionError("server did not become ready")


def chat(base, model, prompt, failure_after=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
    }
    if failure_after is not None:
        payload["vbuf_source_failure_after_requests"] = failure_after
    status, body = request(base, payload)
    content = ""
    if status == 200:
        content = body["choices"][0]["message"]["content"]
    return {
        "status": status,
        "content": content,
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": body.get("usage", {}).get("completion_tokens", 0),
        "error": body.get("error", {}).get("message", ""),
    }


def server_run(args, port, enable_faults, actions):
    log_file = tempfile.NamedTemporaryFile(prefix="vbuf-step31u-", suffix=".log", mode="w+")
    command = [
        args.server_binary,
        "--semantic-model", args.semantic_model,
        "--source-url", args.source_url,
        "--model-alias", args.model,
        "--host", args.host,
        "--port", str(port),
        "--blocks", str(args.blocks),
        "--capacity", str(args.capacity if not enable_faults else args.fault_capacity),
        "--max-new-tokens", "1",
        "--runtime-mode", "normal",
    ]
    if enable_faults:
        command.append("--enable-qualification-faults")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": args.ld_library_path},
    )
    try:
        base = f"http://{args.host}:{port}"
        wait_ready(base)
        responses = [chat(base, args.model, args.prompt, failure_after) for failure_after in actions]
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=60)
        log_file.flush()
        log_file.seek(0)
    text = log_file.read()
    log_file.close()
    assert process.returncode == 0, f"server exited with {process.returncode}"
    return responses, parse_records(text), text


def run_direct(args):
    command = [
        args.direct_binary,
        "--semantic-model", args.semantic_model,
        "--source-url", args.source_url,
        "--model-alias", args.model,
        "--prompt", args.prompt,
        "--blocks", str(args.blocks),
        "--capacity", str(args.capacity),
        "--max-new-tokens", "1",
        "--warmup", str(args.warmup),
        "--requests", str(args.requests),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=3600,
        env={**os.environ, "LD_LIBRARY_PATH": args.ld_library_path},
    )
    records = parse_records(result.stdout)
    assert len(records) == args.warmup + args.requests, result.stdout
    assert all(record["prompt_token_hash"] == records[0]["prompt_token_hash"] for record in records)
    assert all(record["generated_tokens"] == "1" for record in records)
    assert records[0]["source_bytes"] != "0"
    assert all(record["source_bytes"] == "0" for record in records[1:])
    return records


def qualify_overhead(args):
    runs = []
    all_direct_measure = []
    all_server_measure = []
    for repetition in range(args.repetitions):
        direct_records = run_direct(args)
        responses, server_records, _ = server_run(
            args, args.port + repetition, False, [None] * (args.warmup + args.requests))
        assert [response["status"] for response in responses] == [200] * len(responses)
        output = bytes.fromhex(direct_records[args.warmup]["output"]).decode()
        assert all(response["content"] == output for response in responses)
        assert all(record["prompt_token_hash"] == direct_records[0]["prompt_token_hash"] for record in server_records)
        assert all(record["generated_tokens"] == "1" for record in server_records)
        direct_measure = [int(record["total_ns"]) for record in direct_records[args.warmup:]]
        server_measure = [int(record["total_ns"]) for record in server_records[args.warmup:]]
        all_direct_measure.extend(direct_measure)
        all_server_measure.extend(server_measure)
        runs.append({"direct_records": direct_records, "server_records": server_records})
    direct_median_ns = statistics.median(all_direct_measure)
    server_median_ns = statistics.median(all_server_measure)
    overhead_ns = server_median_ns - direct_median_ns
    overhead_pct = overhead_ns / direct_median_ns * 100.0
    if overhead_pct <= 5.0:
        classification = "small"
    elif overhead_pct <= 15.0:
        classification = "material"
    else:
        classification = "large"
    return {
        "repetitions": args.repetitions,
        "direct_measurements": all_direct_measure,
        "server_measurements": all_server_measure,
        "runs": runs,
        "direct_median_ns": direct_median_ns,
        "server_median_ns": server_median_ns,
        "overhead_ns": overhead_ns,
        "overhead_ms": overhead_ns / 1_000_000.0,
        "overhead_percent": overhead_pct,
        "classification": classification,
    }


def qualify_faults(args):
    actions = [None, 1, None, 30, None, 60, None]
    responses, records, log_text = server_run(args, args.port + 1, True, actions)
    assert [response["status"] for response in responses] == [200, 500, 200, 500, 200, 500, 200]
    failures = [records[index] for index in (1, 3, 5)]
    recoveries = [records[index] for index in (2, 4, 6)]
    assert [record["source_failure_injected"] for record in failures] == ["yes"] * 3
    assert all(record["active_leases_after"] == "0" for record in failures + recoveries)
    assert all(record["active_inflight_bytes_after"] == "0" for record in failures + recoveries)
    assert all(int(record["resident_bytes_after"]) <= args.fault_capacity for record in failures + recoveries)
    progress = [(int(record["completed_layers"]), int(record["completed_positions"])) for record in failures]
    assert progress == [(1, 0), (1, 4), (2, 9)], progress
    assert all(response["content"] for response in responses[::2])
    assert all(response["completion_tokens"] == 1 for response in responses[::2])
    assert log_text.count("source=controlled-failure") >= 3
    return {
        "responses": responses,
        "failure_records": failures,
        "recovery_records": recoveries,
        "progress": progress,
        "clean_recovery": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-binary", required=True)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--model", default="vbuf-step31u")
    parser.add_argument("--prompt", default="Say hi")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--fault-capacity", type=int, default=67108864)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    overhead = qualify_overhead(args)
    faults = qualify_faults(args)
    evidence = {
        "qualification": "Step 31U",
        "model": args.model,
        "prompt": args.prompt,
        "prompt_token_hash": overhead["runs"][0]["direct_records"][0]["prompt_token_hash"],
        "blocks": args.blocks,
        "max_new_tokens": 1,
        "overhead": overhead,
        "fault_matrix": faults,
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(evidence, output, ensure_ascii=False, indent=2)
    print("VBUF_STEP31U_QUALIFICATION=PASS")
    print(f"DIRECT_MEDIAN_MS={overhead['direct_median_ns'] / 1_000_000.0:.3f}")
    print(f"SERVER_MEDIAN_MS={overhead['server_median_ns'] / 1_000_000.0:.3f}")
    print(f"SERVER_OVERHEAD_MS={overhead['overhead_ms']:.3f}")
    print(f"SERVER_OVERHEAD_PERCENT={overhead['overhead_percent']:.3f}")
    print("SERVER_OVERHEAD_CLASSIFICATION=" + overhead["classification"].upper())
    print("FAILURE_PROGRESS=" + json.dumps(faults["progress"]))
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
