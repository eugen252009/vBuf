#!/usr/bin/env python3
"""Step 31X cancellation-aware serial admission qualification."""

import argparse
import json
import re
import signal
import socket
import time
from concurrent.futures import ThreadPoolExecutor

from vbuf_step31v_serial_queue_test import (
    chat,
    read_http_response,
    response_id,
    send_raw,
    start_server,
)
from vbuf_step31w_control_plane_test import control


FIELDS = re.compile(r"([a-z_]+)=([^\s]+)")


def parse_admission_log(log_text):
    admissions = []
    requests = []
    for line in log_text.splitlines():
        if line.startswith("vbuf_admission "):
            admissions.append(dict(FIELDS.findall(line)))
        elif line.startswith("vbuf_request "):
            requests.append(dict(FIELDS.findall(line)))
    return admissions, requests


def stop_server_with_log(process, log):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    log.flush()
    log.seek(0)
    text = log.read()
    log.close()
    assert process.returncode == 0, f"server exited with {process.returncode}"
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
    admissions, request_logs = parse_admission_log(text)
    return {
        "records": records,
        "controls": controls,
        "connections": connections,
        "admissions": admissions,
        "request_logs": request_logs,
    }


def wait_for_waiting(log, count, orders=None, timeout=10):
    deadline = time.monotonic() + timeout
    waiting = []
    while time.monotonic() < deadline:
        log.flush()
        log.seek(0)
        text = log.read()
        log.seek(0, 2)
        waiting = [item for item in parse_admission_log(text)[0]
            if item.get("event") == "waiting" and "connection_order" in item]
        observed_orders = {int(item["connection_order"]) for item in waiting}
        if len(waiting) >= count and (orders is None or set(orders).issubset(observed_orders)):
            return waiting
        time.sleep(0.02)
    raise AssertionError(
        f"only observed {len(waiting)} waiting admissions, expected {count}; "
        f"orders={[item.get('connection_order') for item in waiting]}"
    )


def raw_generation(args, port, max_tokens=1, stream=False):
    client = socket.create_connection((args.host, port), timeout=30)
    client.settimeout(900)
    send_raw(client, args.model, max_tokens, stream=stream)
    return client


def run_single_queued_cancel(args, port, stream=False):
    process, log, base = start_server(args, port, max_tokens=8)
    queued = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            queued = raw_generation(args, port, 1, stream=stream)
            wait_for_waiting(log, 1, orders=(2,))
            queued.close()
            queued = None
            active = active_future.result()
            recovery = pool.submit(chat, base, args.model, 1).result()
    finally:
        if queued is not None:
            queued.close()
        evidence = stop_server_with_log(process, log)
    assert active["status"] == recovery["status"] == 200
    waiting = [item for item in evidence["admissions"] if item["event"] == "waiting"]
    cancelled = [item for item in evidence["admissions"] if item["event"] == "terminal" and item["admission_state"] == "CANCELLED"]
    assert any(item["id"] == cancelled[0]["id"] for item in waiting)
    assert len(cancelled) == 1
    item = cancelled[0]
    assert item["runtime_entry_count"] == "0"
    assert item["inference_lease_acquires"] == "0"
    assert not any(record["id"] == item["id"] for record in evidence["records"])
    assert len(evidence["records"]) == 2
    return {
        "active": active,
        "recovery": recovery,
        "stream": stream,
        "cancelled_admission": item,
        "runtime_entry_count": int(item["runtime_entry_count"]),
        "lease_acquires": int(item["inference_lease_acquires"]),
        **evidence,
    }


def run_waiter_hole(args, port, cancelled_index):
    process, log, base = start_server(args, port, max_tokens=8)
    sockets = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            for _ in range(3):
                sockets.append(raw_generation(args, port, 1))
                time.sleep(0.2)
            wait_for_waiting(log, 1, orders=(2 + cancelled_index,))
            sockets[cancelled_index].close()
            sockets[cancelled_index] = None
            active = active_future.result()
            responses = []
            for index, client in enumerate(sockets):
                if client is None:
                    continue
                body = read_http_response(client)
                responses.append((index, response_id(body), int(body.split(b" ", 2)[1])))
                client.close()
    finally:
        for client in sockets:
            if client is not None:
                client.close()
        evidence = stop_server_with_log(process, log)
    assert active["status"] == 200
    assert all(status == 200 for _, _, status in responses)
    cancelled = [item for item in evidence["admissions"] if item["event"] == "terminal" and item["admission_state"] == "CANCELLED"]
    assert len(cancelled) == 1 and cancelled[0]["runtime_entry_count"] == "0"
    execution = sorted(
        [record for record in evidence["records"] if record["id"] != active["id"]],
        key=lambda record: int(record["request_index"]),
    )
    expected_survivors = [response_id for index, response_id, _ in responses]
    assert [record["id"] for record in execution] == expected_survivors
    assert not any(record["id"] == cancelled[0]["id"] for record in evidence["records"])
    return {
        "cancelled_index": cancelled_index,
        "execution_order": [record["id"] for record in execution],
        "surviving_socket_order": [index for index, _, _ in responses],
        "cancelled_admission": cancelled[0],
        **evidence,
    }


def run_all_waiters_cancel(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    sockets = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            sockets = []
            for _ in range(3):
                sockets.append(raw_generation(args, port, 1))
                time.sleep(0.2)
            wait_for_waiting(log, 3, orders=(2, 3, 4))
            for client in sockets:
                client.close()
            sockets = []
            active = active_future.result()
            next_request = pool.submit(chat, base, args.model, 1).result()
    finally:
        for client in sockets:
            client.close()
        evidence = stop_server_with_log(process, log)
    cancelled = [item for item in evidence["admissions"] if item["event"] == "terminal" and item["admission_state"] == "CANCELLED"]
    assert len(cancelled) == 3
    assert all(item["runtime_entry_count"] == "0" for item in cancelled)
    assert active["status"] == next_request["status"] == 200
    assert len(evidence["records"]) == 2
    return {"cancelled_count": len(cancelled), "active": active, "next": next_request, **evidence}


def run_failure_cancel_follower(args, port):
    process, log, base = start_server(args, port, max_tokens=8, capacity=67108864, faults=True)
    cancelled_socket = None
    follower_socket = None
    try:
        warmup = chat(base, args.model, 1)
        assert warmup["status"] == 200
        pre_failure = chat(base, args.model, 1, False, 1)
        pre_recovery = chat(base, args.model, 1)
        assert pre_failure["status"] == 500 and pre_recovery["status"] == 200
        with ThreadPoolExecutor(max_workers=1) as pool:
            failure_future = pool.submit(chat, base, args.model, 8, False, 30)
            time.sleep(0.2)
            cancelled_socket = raw_generation(args, port, 1)
            follower_socket = raw_generation(args, port, 1)
            wait_for_waiting(log, 2, orders=(5, 6))
            cancelled_socket.close()
            cancelled_socket = None
            failure = failure_future.result()
            follower_body = read_http_response(follower_socket)
            follower_socket.close()
            follower_socket = None
    finally:
        if cancelled_socket is not None:
            cancelled_socket.close()
        if follower_socket is not None:
            follower_socket.close()
        evidence = stop_server_with_log(process, log)
    assert failure["status"] == 500
    assert int(follower_body.split(b" ", 2)[1]) == 200
    cancelled = [item for item in evidence["admissions"] if item["event"] == "terminal" and item["admission_state"] == "CANCELLED"]
    assert len(cancelled) == 1 and cancelled[0]["runtime_entry_count"] == "0"
    follower_id = response_id(follower_body)
    assert any(record["id"] == follower_id for record in evidence["records"])
    assert not any(record["id"] == cancelled[0]["id"] for record in evidence["records"])
    return {"failure": failure, "follower_id": follower_id, **evidence}


def run_control_with_waiters(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    waiters = []
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active_future = pool.submit(chat, base, args.model, 8)
            time.sleep(0.2)
            waiters = [raw_generation(args, port, 1)]
            time.sleep(0.2)
            waiters.append(raw_generation(args, port, 1))
            wait_for_waiting(log, 2, orders=(2, 3))
            health = control(base, "/health")
            models = control(base, "/v1/models")
            for client in waiters:
                client.close()
            waiters = []
            active = active_future.result()
    finally:
        for client in waiters:
            client.close()
        evidence = stop_server_with_log(process, log)
    assert health["status"] == models["status"] == active["status"] == 200
    assert health["finished_ns"] < active["finished_ns"]
    assert models["finished_ns"] < active["finished_ns"]
    return {"health": health, "models": models, **evidence}


def run_shutdown_with_waiters(args, port):
    process, log, base = start_server(args, port, max_tokens=8)
    active = None
    waiters = []
    try:
        active = raw_generation(args, port, 8, stream=True)
        time.sleep(0.2)
        waiters = [raw_generation(args, port, 1)]
        time.sleep(0.2)
        waiters.append(raw_generation(args, port, 1))
        wait_for_waiting(log, 2, orders=(2, 3))
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=60)
    finally:
        if active is not None:
            active.close()
        for client in waiters:
            client.close()
    log.flush()
    log.seek(0)
    text = log.read()
    log.close()
    assert process.returncode == 0
    admissions, requests = parse_admission_log(text)
    cancelled = [item for item in admissions if item["event"] == "terminal" and item["admission_state"] == "CANCELLED"]
    waiting_cancelled = [item for item in cancelled if item["runtime_entry_count"] == "0"]
    assert len(waiting_cancelled) >= 2
    return {"admissions": admissions, "request_logs": requests, "clean_shutdown": True}


def run_admission_race(args, port):
    outcomes = []
    for repetition in range(3):
        process, log, base = start_server(args, port + repetition, max_tokens=8)
        queued = None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                active_future = pool.submit(chat, base, args.model, 8)
                time.sleep(0.2)
                queued = raw_generation(args, port + repetition, 1, stream=True)
                wait_for_waiting(log, 1, orders=(2,))
                while not active_future.done():
                    time.sleep(0.001)
                active = active_future.result()
                queued.close()
                queued = None
        finally:
            if queued is not None:
                queued.close()
            evidence = stop_server_with_log(process, log)
        admissions = [item for item in evidence["admissions"] if item["event"] == "terminal"]
        assert len(admissions) == 2
        queued_admission = [item for item in admissions if item["id"] != active["id"]][0]
        runtime_records = [record for record in evidence["records"] if record["id"] == queued_admission["id"]]
        if queued_admission["runtime_entry_count"] == "0":
            assert not runtime_records
            outcome = "CANCELLED_BEFORE_ADMISSION"
        else:
            assert queued_admission["runtime_entry_count"] == "1"
            assert len(runtime_records) == 1
            outcome = "ADMITTED_BEFORE_CANCELLATION"
        outcomes.append(outcome)
    return {"outcomes": outcomes, "single_winner": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--model", default="vbuf-step31x")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18740)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    results = {
        "queued_disconnect_non_stream": run_single_queued_cancel(args, args.port),
        "queued_disconnect_stream": run_single_queued_cancel(args, args.port + 1, stream=True),
        "middle_cancel": run_waiter_hole(args, args.port + 2, 1),
        "first_cancel": run_waiter_hole(args, args.port + 3, 0),
        "last_cancel": run_waiter_hole(args, args.port + 4, 2),
        "all_cancel": run_all_waiters_cancel(args, args.port + 5),
        "failure_cancel_follower": run_failure_cancel_follower(args, args.port + 6),
        "control_with_waiters": run_control_with_waiters(args, args.port + 7),
        "shutdown_with_waiters": run_shutdown_with_waiters(args, args.port + 8),
        "admission_race": run_admission_race(args, args.port + 9),
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print("VBUF_STEP31X_CANCELLABLE_ADMISSION_QUALIFICATION=PASS")
    print("QUEUED_DISCONNECT_NON_STREAM=PASS")
    print("QUEUED_DISCONNECT_STREAM=PASS")
    print("MIDDLE_CANCEL=PASS")
    print("FIRST_CANCEL=PASS")
    print("LAST_CANCEL=PASS")
    print("ALL_CANCEL=PASS")
    print("FAILURE_CANCEL_FOLLOWER=PASS")
    print("CONTROL_WITH_WAITERS=PASS")
    print("SHUTDOWN_WITH_WAITERS=PASS")
    print("ADMISSION_RACE=PASS")
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
