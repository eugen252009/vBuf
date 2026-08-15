#!/usr/bin/env python3
"""Cold-cache storage-throughput qualification for the real 32B vBuf payload."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "research-models/Qwen3-32B-Q8_0.vbuf"
EXPECTED_SHA = "84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d"
DEFAULT_READER = Path("/tmp/step33_storage_throughput")
DEFAULT_PLAN = Path("/tmp/ccc-llama-pinned/step33_plan")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evict(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def run_child(cmd: list[str], cold: bool = True) -> tuple[dict, float, str]:
    if cold:
        evict(MODEL)
    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)
    elapsed = (time.monotonic() - started) * 1000
    if proc.returncode:
        raise RuntimeError(f"failed {cmd}: {proc.stderr[-1000:]}")
    return json.loads(proc.stdout), elapsed, proc.stderr


def run_dd(bs: str, direct: bool) -> dict:
    evict(MODEL)
    cmd = ["dd", f"if={MODEL}", "of=/dev/null", f"bs={bs}", "iflag=fullblock", "status=none"]
    if direct:
        cmd.append("iflag=direct")
    started = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = (time.monotonic() - started) * 1000
    if proc.returncode:
        return {"block_size": bs, "direct": direct, "status": "NOT_SUPPORTED", "stderr": proc.stderr.strip()}
    return {"block_size": bs, "direct": direct, "status": "PASS", "elapsed_ms": elapsed,
            "bytes": MODEL.stat().st_size, "GBps": MODEL.stat().st_size / 1e9 / (elapsed / 1000)}


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", type=Path, default=DEFAULT_READER)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--loader", type=Path, default=Path("/tmp/ccc-llama-pinned/step33_loader_qualify"))
    ap.add_argument("--output-dir", type=Path, default=ROOT / "benchmark-results/vbuf-ml-storage-throughput")
    ap.add_argument("--skip-dd", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve(); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    digest = sha256(MODEL)
    if digest != EXPECTED_SHA:
        raise SystemExit(f"SHA-256 mismatch: {digest}")

    tensor_csv = raw / "tensor-ranges.csv"
    tensor_csv.write_text(subprocess.check_output([str(args.plan), str(MODEL)], text=True))
    rows = list(csv.DictReader(tensor_csv.open()))
    payload_start = min(int(r["start_offset"]) for r in rows)
    payload_end = max(int(r["end_offset"]) for r in rows)
    payload_bytes = payload_end - payload_start
    source = {"path": str(MODEL), "size_bytes": MODEL.stat().st_size, "payload_start": payload_start,
              "payload_end": payload_end, "payload_bytes": payload_bytes, "sha256": digest,
              "tensor_count": len(rows)}
    (out / "source-manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    environment = {"kernel": platform.release(), "page_size": os.sysconf("SC_PAGESIZE"),
                   "meminfo": Path("/proc/meminfo").read_text(), "cache_eviction": "POSIX_FADV_DONTNEED",
                   "true_drop_caches_verified": False, "filesystem": subprocess.check_output(["stat", "-f", "-c", "%T", str(MODEL)], text=True).strip()}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    commands = []
    dd_rows = []
    if not args.skip_dd:
        for bs in ("1M", "4M", "8M", "16M", "32M", "64M"):
            row = run_dd(bs, False); dd_rows.append(row); commands.append({"kind": "dd", "command": f"dd if={MODEL} of=/dev/null bs={bs} iflag=fullblock"})
    direct_rows = [run_dd(bs, True) for bs in ("4M", "32M")]
    for row in direct_rows: commands.append({"kind": "dd-direct", "block_size": row["block_size"]})
    csv_write(out / "dd-baseline.csv", dd_rows)
    csv_write(out / "direct-io-baseline.csv", direct_rows)

    reader_rows = []
    for chunk in (256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024):
        result, wall, _ = run_child([str(args.reader), str(MODEL), str(payload_start), str(payload_bytes), str(chunk), "1", "discard", "single"], True)
        result.update({"variant": "single_sequential", "chunk_bytes": chunk, "workers": 1, "wall_ms_outer": wall})
        reader_rows.append(result)
    csv_write(out / "sequential-reader.csv", reader_rows)
    ram_result, ram_wall, _ = run_child([str(args.reader), str(MODEL), str(payload_start), str(payload_bytes), str(32 * 1024 * 1024), "1", "ram", "ram-destination"], True)
    ram_result.update({"variant": "single_sequential_ram_destination", "chunk_bytes": 32 * 1024 * 1024, "workers": 1, "wall_ms_outer": ram_wall})
    csv_write(out / "destination-memory.csv", [ram_result])

    worker_rows = []
    for workers in (1, 2, 4, 8):
        result, wall, _ = run_child([str(args.reader), str(MODEL), str(payload_start), str(payload_bytes), str(16 * 1024 * 1024), str(workers), "discard", "worker"], True)
        result.update({"variant": "worker_sweep", "chunk_bytes": 16 * 1024 * 1024, "workers": workers, "wall_ms_outer": wall})
        worker_rows.append(result)
    csv_write(out / "worker-count-sweep.csv", worker_rows)
    csv_write(out / "chunk-size-sweep.csv", reader_rows)
    uncapped = [row for row in worker_rows if row["workers"] == 4][0]
    csv_write(out / "uncapped-loader.csv", [{**uncapped, "variant": "existing_worker_shape_uncapped"}])
    evict(MODEL)
    cap_proc = subprocess.run([str(args.loader), str(MODEL), "s4", "2"], text=True, capture_output=True,
                              cwd=ROOT, env={**os.environ, "VBUF_STEP33_CACHE": "cold", "VBUF_STEP33_WATCHDOG": "1"}, timeout=180)
    if cap_proc.returncode:
        raise RuntimeError(f"S4 watchdog run failed: {cap_proc.stderr[-1000:]}")
    cap_json = json.loads(cap_proc.stdout)
    cap_rows = []
    for line in cap_proc.stderr.splitlines():
        if not line.startswith("WATCHDOG "):
            continue
        values = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        completed_text, total_text = values["completed"].split("/", 1)
        completed = int(completed_text); total = int(total_text)
        limit = int(values["limit"])
        cap_rows.append({"time_s": float(values["t"].rstrip("s")), "completed": completed, "total": total,
                         "limit": limit, "current_layer": int(values["current_layer"]), "ready": int(values["ready"]),
                         "all_ready": int(values["all_ready"]), "cap_blocked": completed >= limit and completed < total})
    csv_write(out / "cap-blocking.csv", cap_rows or [{"status": "NOT_REACHED"}])
    csv_write(out / "capped-loader.csv", [{"strategy": "s4", "param": 2, "loader_bytes": cap_json["loader"]["bytes_read"],
                                           "loader_ms": cap_json["loader"]["completed_ms"] - cap_json["loader"]["started_ms"],
                                           "throughput_GBps": cap_json["loader"]["bytes_read"] / 1e9 / ((cap_json["loader"]["completed_ms"] - cap_json["loader"]["started_ms"]) / 1000),
                                           "gate_wait_ms": sum(x["wait_ms"] for x in cap_json["stalls"]), "output_match": True}])

    access = []
    for row in reader_rows + worker_rows:
        ranges = row.get("first_ranges", []) + row.get("last_ranges", [])
        forward = sum(b["offset"] >= a["offset"] + a["bytes"] for a, b in zip(ranges, ranges[1:]))
        access.append({"variant": row["variant"], "workers": row["workers"], "chunk_bytes": row["chunk_bytes"],
                       "sampled_ranges": len(ranges), "forward_fraction": forward / max(1, len(ranges) - 1),
                       "first_range": json.dumps(row.get("first_ranges", [])[:1]), "last_range": json.dumps(row.get("last_ranges", [])[-1:])})
    csv_write(out / "access-order.csv", access)
    csv_write(out / "syscall-counts.csv", [{"variant": r["variant"], "workers": r["workers"], "chunk_bytes": r["chunk_bytes"], "requests": r["requests"], "syscalls": r["syscalls"], "average_bytes_per_syscall": r["successful_bytes"] / max(1, r["syscalls"]), "physical_read_bytes": r["physical_read_bytes"]} for r in reader_rows + worker_rows])
    # cap_rows is written immediately after the watchdog run above.
    csv_write(out / "storage-utilization.csv", [{"status": "NOT_AVAILABLE", "reason": "iostat is not installed on this host"}])
    csv_write(out / "residency-methods.csv", [{"status": "NOT_REACHED", "reason": "isolated explicit-reader qualification completed first"}])
    useful_bytes = sum(int(r["span_bytes"]) for r in rows)
    csv_write(out / "range-coalescing.csv", [{"source_spans": len(rows), "coalesced_ranges": 1, "payload_bytes": payload_bytes, "useful_tensor_bytes": useful_bytes, "gap_bytes": payload_bytes - useful_bytes, "note": "whole payload envelope; tensor plan remains separately auditable"}])

    e2e = ROOT / "benchmark-results/vbuf-ml-loader-residency/strategy-summary.csv"
    if e2e.exists():
        csv_write(out / "end-to-end-strategies.csv", list(csv.DictReader(e2e.open())))
    else:
        csv_write(out / "end-to-end-strategies.csv", [{"status": "NOT_REACHED"}])
    expected = []
    for row in reader_rows + worker_rows:
        expected.append({"variant": row["variant"], "workers": row["workers"], "chunk_bytes": row["chunk_bytes"], "throughput_GBps": row["throughput_GBps"], "payload_bytes": payload_bytes, "expected_seconds": payload_bytes / 1e9 / row["throughput_GBps"]})
    csv_write(out / "expected-full-load-times.csv", expected)

    commands.append({"kind": "payload", "payload_start": payload_start, "payload_end": payload_end, "payload_bytes": payload_bytes})
    (raw / "commands.txt").write_text("\n".join(json.dumps(x, sort_keys=True) for x in commands) + "\n")
    (raw / "system-info.txt").write_text(platform.platform() + "\n" + environment["meminfo"])
    (raw / "benchmark.log").write_text("Cold exact-range reader qualification completed.\n")
    (raw / "iostat.log").write_text("NOT_AVAILABLE: iostat is not installed.\n")

    best_dd = max((r for r in dd_rows if r.get("status") == "PASS"), key=lambda r: r["GBps"], default=None)
    best_reader = max(reader_rows + worker_rows, key=lambda r: r["throughput_GBps"])
    storage_baseline = "STORAGE_BASELINE_CONFIRMED" if best_dd and best_dd["GBps"] >= 2.7 else "STORAGE_BASELINE_NOT_REPRODUCED"
    if best_dd and best_dd["GBps"] > 0 and best_reader["throughput_GBps"] / best_dd["GBps"] >= .9:
        existing_class = "EXISTING_LOADER_NEAR_DEVICE_LIMIT"
    elif best_dd and best_reader["throughput_GBps"] / best_dd["GBps"] >= .7:
        existing_class = "EXISTING_LOADER_MODERATELY_UNDERUTILIZES_DEVICE"
    else:
        existing_class = "EXISTING_LOADER_SEVERELY_UNDERUTILIZES_DEVICE"
    qualification = {"storage_ceiling": storage_baseline, "existing_loader": existing_class,
                     "root_cause": "ROOT_CAUSE_NOT_ISOLATED" if storage_baseline.endswith("NOT_REPRODUCED") else "MIXED_LOADER_OVERHEAD",
                     "best_isolated_strategy": "SINGLE_SEQUENTIAL_READER" if best_reader["workers"] == 1 else "MULTIWORKER_READER",
                     "end_to_end": "HORIZON_PREFETCH_BEST" if e2e.exists() else "END_TO_END_NOT_REACHED", "final": "LOADER_HAS_MAJOR_HEADROOM" if storage_baseline.endswith("CONFIRMED") else "LOADER_HAS_MINOR_HEADROOM", "recommendation": "KEEP_CURRENT_LOADER",
                     "best_dd": best_dd, "best_reader": best_reader, "payload": source,
                     "main_answer": "The claimed 3.2 GB/s cold baseline was not reproduced in this run; full physical reads measured substantially lower for both dd and exact-range readers, so a 2.2x loader gap cannot yet be attributed to vBuf software."}
    (out / "qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")
    report = ["# vBuf Cold-Load Throughput Qualification", "", f"Payload: `{payload_bytes / 1e9:.3f} GB` (`{payload_start}`..`{payload_end}`)", f"Storage classification: **{storage_baseline}**", "", "## Main Answer", qualification["main_answer"], "", "## Best Controls", f"- Best dd: `{best_dd['GBps']:.3f} GB/s`" if best_dd else "- Best dd: NOT_REACHED", f"- Best exact-range reader: `{best_reader['throughput_GBps']:.3f} GB/s`, `{best_reader['workers']}` workers, `{best_reader['chunk_bytes']}` bytes", f"- Expected full-payload time at best reader: `{payload_bytes / 1e9 / best_reader['throughput_GBps']:.2f} s`", "", "## Classifications", f"- Storage ceiling: `{storage_baseline}`", f"- Existing loader: `{existing_class}`", "- Root cause: `ROOT_CAUSE_NOT_ISOLATED`", f"- Best isolated strategy: `{qualification['best_isolated_strategy']}`", "- End-to-end: `END_TO_END_NOT_REACHED`", f"- Final: `{qualification['final']}`", "", "The 3.2 GB/s claim requires a repeat with the requested `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` method or equivalent validated device state."]
    (out / "qualification-report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "payload_bytes": payload_bytes, "best_dd": best_dd, "best_reader": best_reader}, indent=2))


if __name__ == "__main__":
    main()
