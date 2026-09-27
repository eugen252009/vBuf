#!/usr/bin/env python3
"""Measure the existing full-model server inside an externally limited cgroup.

Run through systemd-run with MemoryMax and MemorySwapMax=0. The cgroup includes
the runtime, loopback range provider, client, and sampler. This is a research
driver, not a new runtime/source implementation. It does not establish parity.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from typing import cast
import urllib.request

from range_server import RangeHandler, RangeServer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize():
    parser = argparse.ArgumentParser(description="Preserve complete and OOM-interrupted run evidence")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    samples = [json.loads(line) for line in (args.run / "memory.jsonl").read_text().splitlines()]
    result_path = args.run / "result.json"
    evidence = {"run_directory": str(args.run), "unit": args.unit,
                "result": json.loads(result_path.read_text()) if result_path.exists() else None,
                "systemd_properties": subprocess.check_output([
                    "systemctl", "--user", "show", args.unit, "-p", "Result", "-p", "MemoryPeak",
                    "-p", "ExecMainCode", "-p", "ExecMainStatus", "-p", "MemoryMax", "-p", "MemorySwapMax"], text=True),
                "journal": subprocess.check_output([
                    "journalctl", "--user", "-u", args.unit, "--no-pager"], text=True),
                "server_log": (args.run / "server.log").read_text(),
                "stream_events": {path.name: [json.loads(line) for line in path.read_text().splitlines()]
                                  for path in sorted(args.run.glob("request-*.jsonl"))},
                "sample_count": len(samples),
                "sampled_peak_runtime_rss_bytes": max(row["runtime"].get("VmRSS", 0) for row in samples),
                "sampled_peak_runtime_anon_bytes": max(row["runtime"].get("RssAnon", 0) for row in samples),
                "sampled_peak_provider_rss_bytes": max(row["provider_and_sampler"].get("VmRSS", 0) for row in samples),
                "sampled_peak_cgroup_bytes": max(row["memory_current"] for row in samples),
                "sampled_peak_swap_bytes": max(row["runtime"].get("VmSwap", 0) for row in samples),
                "first_sample": samples[0], "last_sample": samples[-1],
                "peak_runtime_sample": max(samples, key=lambda row: row["runtime"].get("VmRSS", 0)),
                "raw_memory_sha256": hashlib.sha256((args.run / "memory.jsonl").read_bytes()).hexdigest()}
    heap_path = args.run / "heap.jsonl"
    if heap_path.exists():
        heap = [json.loads(line) for line in heap_path.read_text().splitlines()]
        evidence["heap"] = {"samples": len(heap),
            "peak_arena_bytes": max(row["arena_bytes"] for row in heap),
            "peak_free_arena_bytes": max(row["arena_free_bytes"] for row in heap),
            "peak_live_malloc_bytes": max(row["arena_used_bytes"] + row["malloc_mmap_bytes"] for row in heap),
            "last_sample": heap[-1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, evidence)
    # Preserve the complete time series alongside the compact summary.
    args.output.with_suffix(".memory.jsonl").write_bytes((args.run / "memory.jsonl").read_bytes())
    if heap_path.exists():
        args.output.with_suffix(".heap.jsonl").write_bytes(heap_path.read_bytes())
    print(json.dumps({key: evidence[key] for key in (
        "unit", "systemd_properties", "sample_count", "sampled_peak_runtime_rss_bytes",
        "sampled_peak_runtime_anon_bytes", "sampled_peak_provider_rss_bytes", "sampled_peak_cgroup_bytes")}))
    return 0


def counters(path):
    try:
        return {key: int(value) for key, value in
                (line.split() for line in path.read_text().splitlines())}
    except (FileNotFoundError, ProcessLookupError):
        return {}


def rss(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return {line.split(":")[0]: int(line.split()[1]) * 1024
                for line in lines if line.startswith(("VmRSS:", "VmHWM:", "RssAnon:", "RssFile:", "VmSwap:"))}
    except (FileNotFoundError, ProcessLookupError):
        return {}


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def stream_request(base, prompt, tokens, timeout, events_path):
    payload = {"model": "deepseek-footprint", "prompt": prompt,
               "max_tokens": tokens, "stream": True}
    request = urllib.request.Request(base + "/v1/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    pieces, arrivals = [], []
    done = False
    finish = None
    with events_path.open("w") as evidence:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                elapsed = time.monotonic() - start
                text = raw[6:].strip().decode("utf-8")
                evidence.write(json.dumps({"seconds": elapsed, "data": text}) + "\n")
                evidence.flush()
                if text == "[DONE]":
                    done = True
                    break
                event = json.loads(text)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                choice = event["choices"][0]
                delta = choice.get("delta", {})
                if "content" in delta and "role" not in delta:
                    pieces.append(delta["content"])
                    arrivals.append(elapsed)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    return {"done": done, "finish_reason": finish, "text": "".join(pieces),
            "token_events": len(arrivals), "arrival_seconds": arrivals,
            "time_to_first_token_seconds": arrivals[0] if arrivals else None,
            "elapsed_seconds": time.monotonic() - start,
            "decode_tokens_per_second": ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else None)}


class MeasuredRangeServer(RangeServer):
    def __init__(self, address):
        super().__init__(address, MeasuredRangeHandler)
        self.connections = 0
        self.ranges = []
        self.measure_lock = threading.Lock()

    def get_request(self):
        result = super().get_request()
        with self.measure_lock:
            self.connections += 1
        return result

    def snapshot(self):
        with self.measure_lock:
            ranges = sorted(self.ranges)
            connections = self.connections
        unique = 0
        end = 0
        for start, stop in ranges:
            unique += max(0, stop - max(start, end))
            end = max(end, stop)
        return {"connections": connections, "successful_requests": len(ranges),
                "requested_bytes": sum(stop - start for start, stop in ranges),
                "unique_requested_bytes": unique}


class MeasuredRangeHandler(RangeHandler):
    def log_message(self, format, *args):
        if format == "range=%s status=206 bytes=%d":
            start, end = map(int, args[0][6:].split("-"))
            server = cast(MeasuredRangeServer, self.server)
            with server.measure_lock:
                server.ranges.append((start, end + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=256 * 1024**2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--source-port", type=int, default=18124)
    parser.add_argument("--server-preload", type=Path, help="Trusted diagnostic preload for server only")
    parser.add_argument("--trim-control", action="store_true", help="Diagnostic glibc trim intervention")
    parser.add_argument("--expected-output", type=Path, help="Separately recorded oracle completion bytes")
    parser.add_argument("--runtime-trace", action="store_true")
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument("--expert-threads", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    relative = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                    if line.startswith("0::"))
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limit = (cgroup / "memory.max").read_text().strip()
    swap_limit = (cgroup / "memory.swap.max").read_text().strip()
    if limit == "max" or swap_limit != "0":
        raise RuntimeError("requires an enforced MemoryMax and MemorySwapMax=0")
    report = {"platform": platform.platform(), "cgroup": str(cgroup),
              "memory_limit_bytes": int(limit), "swap_limit_bytes": 0,
              "scope": "runtime + loopback payload provider + client/sampler; no oracle",
              "cache_state": "payload POSIX_FADV_DONTNEED requested once; advisory, coldness not proven; repeated requests reuse runtime/source",
              "payload_bytes": args.payload.stat().st_size,
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}, "requests": []}
    write_json(args.output / "result.json", {**report, "status": "RUNNING"})
    with args.payload.open("rb") as payload:
        os.posix_fadvise(payload.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    source = MeasuredRangeServer(("127.0.0.1", args.source_port))
    source.path, source.mode, source.log_enabled = str(args.payload), "range", False
    threading.Thread(target=source.serve_forever, daemon=True).start()
    command = [str(args.server), "--semantic-model", str(args.semantic_model),
               "--source-url", f"http://127.0.0.1:{args.source_port}/model.vbuf",
               "--model-alias", "deepseek-footprint", "--port", str(args.port),
               "--blocks", str(args.blocks), "--capacity", str(args.capacity),
               "--max-new-tokens", str(args.tokens), "--runtime-mode", "normal"]
    report["command"] = command
    command.extend(["--expert-workers", str(args.expert_workers), "--expert-threads", str(args.expert_threads)])
    if args.runtime_trace:
        command.append("--runtime-trace")
    stop = threading.Event()
    start = time.monotonic()
    with (args.output / "server.log").open("w") as log:
        server_environment = os.environ.copy()
        if args.server_preload:
            server_environment["LD_PRELOAD"] = str(args.server_preload.resolve())
            server_environment["VBUF_HEAP_PROBE_LOG"] = str((args.output / "heap.jsonl").resolve())
        if args.trim_control:
            if not args.server_preload:
                raise RuntimeError("trim control requires the diagnostic preload")
            server_environment["VBUF_HEAP_TRIM_CONTROL"] = "1"
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=server_environment)
        def sample():
            with (args.output / "memory.jsonl").open("w") as evidence:
                while not stop.is_set():
                    entry = {"seconds": time.monotonic() - start,
                             "runtime": rss(process.pid), "provider_and_sampler": rss(os.getpid()),
                             "memory_current": int((cgroup / "memory.current").read_text()),
                             "memory_stat": counters(cgroup / "memory.stat")}
                    evidence.write(json.dumps(entry) + "\n")
                    evidence.flush()
                    stop.wait(0.25)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        base = f"http://127.0.0.1:{args.port}"
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited at startup: {process.returncode}")
                try:
                    report["initial_health"] = get_json(base + "/health")
                    break
                except (OSError, ValueError):
                    time.sleep(0.25)
            else:
                raise RuntimeError("server readiness timeout")
            for index in range(args.requests):
                source_before = source.snapshot()
                result = stream_request(base, args.prompt, args.tokens, args.timeout,
                                        args.output / f"request-{index + 1}.jsonl")
                result["source_before"] = source_before
                result["source_after"] = source.snapshot()
                result["health_after"] = get_json(base + "/health")
                report["requests"].append(result)
                if args.expected_output:
                    result["oracle_text_match"] = result["text"] == args.expected_output.read_text()
                write_json(args.output / "result.json", report)
                print(json.dumps({"request": index + 1, **result}), flush=True)
                if not result["done"]:
                    raise RuntimeError("stream ended without DONE")
                if result.get("oracle_text_match") is False:
                    raise RuntimeError("completion differs from separately recorded oracle")
                if result["finish_reason"] == "length" and result["token_events"] != args.tokens:
                    raise RuntimeError("stream token count differs from requested bound")
            report["status"] = "COMPLETED"
        except Exception as error:
            report["status"], report["error"] = "FAILED", repr(error)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            report["server_exit_code"] = process.returncode
            source.shutdown()
            source.server_close()
            stop.set()
            sampler.join()
            report["memory_peak_bytes"] = int((cgroup / "memory.peak").read_text())
            report["memory_events"] = counters(cgroup / "memory.events")
            report["memory_swap_peak_bytes"] = int((cgroup / "memory.swap.peak").read_text())
            report["source"] = source.snapshot()
            report["total_wall_seconds"] = time.monotonic() - start
            write_json(args.output / "result.json", report)
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(summarize() if sys.argv[1:2] == ["summarize"] else main())
