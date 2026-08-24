#!/usr/bin/env python3
"""Step 31W control-plane responsiveness and lifecycle qualification."""

import argparse
import json
import signal
import socket
import time
from concurrent.futures import ThreadPoolExecutor

from vbuf_step31v_serial_queue_test import (
    chat,
    http_request,
    send_raw,
    start_server,
    stop_server,
)


def control(base, path):
    status, body, started, finished = http_request(base, path=path, timeout=180)
    return {
        "status": status,
        "body": body.decode(errors="replace"),
        "started_ns": started,
        "finished_ns": finished,
        "latency_ms": (finished - started) / 1_000_000.0,
    }


def clean_generation(body):
    response = json.loads(body)
    choice = response["choices"][0]
    return {
        "content": choice.get("message", {}).get("content", choice.get("text", "")),
        "finish_reason": choice["finish_reason"],
        "usage": response["usage"],
    }


def collect_server(process, log):
    records, controls, connections = stop_server(process, log)
    return {"records": records, "controls": controls, "connections": connections}


def run_control_during_generation(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            generation_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            health_future = pool.submit(control, base, "/health")
            models_future = pool.submit(control, base, "/v1/models")
            generation = generation_future.result()
            health = health_future.result()
            models = models_future.result()
    finally:
        evidence = collect_server(process, log)
    assert generation["status"] == health["status"] == models["status"] == 200
    assert health["finished_ns"] < generation["finished_ns"]
    assert models["finished_ns"] < generation["finished_ns"]
    model_list = json.loads(models["body"])
    assert model_list["data"][0]["id"] == args.model
    return {
        "generation": generation,
        "health": health,
        "models": models,
        "health_before_generation_complete": True,
        "models_before_generation_complete": True,
        **evidence,
    }


def run_control_stress(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            generation_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            control_futures = [pool.submit(control, base, "/health") for _ in range(20)]
            control_futures += [pool.submit(control, base, "/v1/models") for _ in range(20)]
            controls = [future.result() for future in control_futures]
            generation = generation_future.result()
    finally:
        evidence = collect_server(process, log)
    assert generation["status"] == 200
    assert all(item["status"] == 200 for item in controls)
    return {**evidence, "generation": generation, "controls": controls}


def run_generation_parity(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        plain = chat(base, args.model, 8)
    finally:
        collect_server(process, log)

    process, log, base = start_server(args, port + 1, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            generation_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            controls = [pool.submit(control, base, "/health") for _ in range(2)]
            controls += [pool.submit(control, base, "/v1/models") for _ in range(2)]
            with_controls = generation_future.result()
            control_results = [future.result() for future in controls]
    finally:
        evidence = collect_server(process, log)
    assert plain["status"] == with_controls["status"] == 200
    assert all(item["status"] == 200 for item in control_results)
    assert clean_generation(plain["body"]) == clean_generation(with_controls["body"])
    return {
        "without_control": clean_generation(plain["body"]),
        "with_control": clean_generation(with_controls["body"]),
        "control_results": control_results,
        "generation_parity": True,
        **evidence,
    }


def run_stream_control(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            stream_future = pool.submit(chat, base, args.model, 8, True)
            time.sleep(0.2)
            health_future = pool.submit(control, base, "/health")
            models_future = pool.submit(control, base, "/v1/models")
            stream = stream_future.result()
            health = health_future.result()
            models = models_future.result()
    finally:
        evidence = collect_server(process, log)
    assert stream["status"] == health["status"] == models["status"] == 200
    assert b"data: [DONE]" in stream["body"].encode()
    assert health["finished_ns"] < stream["finished_ns"]
    assert models["finished_ns"] < stream["finished_ns"]
    return {"stream": stream, "health": health, "models": models, "stream_integrity": True, **evidence}


def run_cancel_control(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    active_socket = socket.create_connection((args.host, port), timeout=30)
    active_socket.settimeout(900)
    try:
        send_raw(active_socket, args.model, 8, stream=True)
        time.sleep(0.2)
        health = control(base, "/health")
        models = control(base, "/v1/models")
        assert health["status"] == models["status"] == 200
        active_socket.close()
        active_socket = None
        recovery = chat(base, args.model, 1)
    finally:
        if active_socket is not None:
            active_socket.close()
        evidence = collect_server(process, log)
    assert recovery["status"] == 200
    cancelled = [record for record in evidence["records"] if record.get("cancelled") == "yes"]
    assert cancelled
    return {"health": health, "models": models, "recovery": recovery, "active_cancelled": True, **evidence}


def run_source_failure(args, port):
    process, log, base = start_server(args, port, max_tokens=8, faults=True)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            failure_future = pool.submit(chat, base, args.model, 8, False, 1)
            time.sleep(0.1)
            health_future = pool.submit(control, base, "/health")
            models_future = pool.submit(control, base, "/v1/models")
            failure = failure_future.result()
            health = health_future.result()
            models = models_future.result()
        recovery = chat(base, args.model, 1)
    finally:
        evidence = collect_server(process, log)
    assert failure["status"] == 500
    assert health["status"] == models["status"] == recovery["status"] == 200
    return {"failure": failure, "health": health, "models": models, "recovery": recovery, **evidence}


def run_serial_generations(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.05)
            second_future = pool.submit(chat, base, args.model, 1)
            first = first_future.result()
            second = second_future.result()
    finally:
        evidence = collect_server(process, log)
    by_id = {record["id"]: record for record in evidence["records"]}
    first_record = by_id[first["id"]]
    second_record = by_id[second["id"]]
    first_end = int(first_record["runtime_end_ns"])
    second_start = int(second_record["runtime_start_ns"])
    assert first["status"] == second["status"] == 200
    assert int(first_record["request_index"]) < int(second_record["request_index"])
    assert first_end <= second_start
    return {
        "first_runtime": [int(first_record["runtime_start_ns"]), first_end],
        "second_runtime": [second_start, int(second_record["runtime_end_ns"])],
        "runtime_intervals_overlap": False,
        "max_simultaneous_active_generations": 1,
        **evidence,
    }


def run_shutdown(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    active_socket = socket.create_connection((args.host, port), timeout=30)
    active_socket.settimeout(30)
    try:
        send_raw(active_socket, args.model, 8, stream=True)
        time.sleep(0.2)
        assert control(base, "/health")["status"] == 200
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    finally:
        active_socket.close()
    log.flush()
    log.seek(0)
    text = log.read()
    log.close()
    assert process.returncode == 0
    assert "vbuf-compat-server stopped cleanly" in text
    return {"returncode": process.returncode, "clean_shutdown": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--model", default="vbuf-step31w")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18340)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    results = {
        "control_during_generation": run_control_during_generation(args, args.port),
        "control_stress": run_control_stress(args, args.port + 1),
        "generation_parity": run_generation_parity(args, args.port + 2),
        "stream_control": run_stream_control(args, args.port + 4),
        "cancel_control": run_cancel_control(args, args.port + 5),
        "source_failure": run_source_failure(args, args.port + 6),
        "serial_generations": run_serial_generations(args, args.port + 7),
        "shutdown": run_shutdown(args, args.port + 8),
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)

    controls = results["control_stress"]["controls"]
    health = [item["latency_ms"] for item in controls if '"status":"ok"' in item["body"]]
    models = [item["latency_ms"] for item in controls if '"object":"list"' in item["body"]]
    print("VBUF_STEP31W_CONTROL_PLANE_QUALIFICATION=PASS")
    print("HEALTH_DURING_GENERATION=PASS")
    print("MODELS_DURING_GENERATION=PASS")
    print("CONTROL_STRESS=PASS")
    print("STREAM_CONTROL=PASS")
    print("CANCELLATION_CONTROL=PASS")
    print("SOURCE_FAILURE_CONTROL=PASS")
    print("SERIAL_GENERATIONS=PASS")
    print("SHUTDOWN=PASS")
    print("HEALTH_STRESS_COUNT=" + str(len(health)))
    print("MODELS_STRESS_COUNT=" + str(len(models)))
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
