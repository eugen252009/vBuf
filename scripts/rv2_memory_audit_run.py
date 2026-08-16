#!/usr/bin/env python3
"""Capture phase-boundary Linux memory evidence for the qualified RV2 bench."""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import pathlib
import re
import subprocess
import time


PAGE = os.sysconf("SC_PAGESIZE")
EVENT_RE = re.compile(r"^EVENT,([^,]+),([^,]+),([^,]+),([^,]+),(.*)$")
MAP_RE = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+([-rwxps]+)\s+([0-9a-f]+)\s+([^ ]+)\s+([0-9]+)\s*(.*)$")
libc = ctypes.CDLL(None, use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.mincore.restype = ctypes.c_int


def read_proc(pid: int, name: str) -> str:
    return pathlib.Path(f"/proc/{pid}/{name}").read_text()


def parse_rollup(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key.strip()):
            continue
        fields = value.strip().split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    return values


def parse_status(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key.strip()):
            continue
        fields = value.strip().split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    return values


def parse_smaps(text: str) -> list[dict]:
    rows = []
    current = None
    for line in text.splitlines():
        match = MAP_RE.match(line)
        if match:
            if current:
                rows.append(current)
            start, end, perms, offset, dev, inode, path = match.groups()
            current = {
                "start": int(start, 16), "end": int(end, 16),
                "size_bytes": int(end, 16) - int(start, 16),
                "perms": perms, "offset": int(offset, 16), "dev": dev,
                "inode": int(inode), "path": path.strip(),
            }
            continue
        if current and ":" in line:
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if fields and fields[0].isdigit():
                current[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
            elif key == "VmFlags":
                current[key] = value.strip()
    if current:
        rows.append(current)
    return rows


def resident_pages(start: int, size: int) -> tuple[int | None, str]:
    aligned = start - (start % PAGE)
    pages = (size + (start - aligned) + PAGE - 1) // PAGE
    total = 0
    chunk_pages = (256 * 1024 * 1024) // PAGE
    for page_offset in range(0, pages, chunk_pages):
        count = min(chunk_pages, pages - page_offset)
        vector = (ctypes.c_ubyte * count)()
        address = aligned + page_offset * PAGE
        if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(count * PAGE), vector) != 0:
            return None, os.strerror(ctypes.get_errno())
        total += sum(byte & 1 for byte in vector)
    return total, "ok"


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["gguf", "vbuf"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binary", default="/home/eugen/rv2_gguf_vbuf_bench")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.csv"
    log = out / f"{args.kind}.log"
    trace.unlink(missing_ok=True)
    command = [args.binary, args.kind, args.model]
    env = os.environ.copy()
    env["VBUF_BENCH_TRACE"] = str(trace)
    (out / "commands.txt").write_text(" ".join(command) + "\n")
    (out / "environment.json").write_text(json.dumps({"command": command, "page_size": PAGE, "pid": None}, indent=2) + "\n")

    with log.open("w") as log_stream:
        process = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT, env=env)
        (out / "environment.json").write_text(json.dumps({"command": command, "page_size": PAGE, "pid": process.pid}, indent=2) + "\n")
        seen = set()
        phase_rows = []
        mapping_rows = []
        residency_rows = []
        while process.poll() is None:
            try:
                trace_lines = trace.read_text().splitlines()
            except FileNotFoundError:
                trace_lines = []
            for line in trace_lines:
                match = EVENT_RE.match(line)
                if not match:
                    continue
                event, timestamp, event_pid, tid, detail = match.groups()
                if event in seen:
                    continue
                seen.add(event)
                try:
                    status_text = read_proc(process.pid, "status")
                    rollup_text = read_proc(process.pid, "smaps_rollup")
                    smaps_text = read_proc(process.pid, "smaps")
                    maps_text = read_proc(process.pid, "maps")
                except (FileNotFoundError, ProcessLookupError):
                    continue
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", event)
                (out / f"{args.kind}-smaps-{safe}.txt").write_text(smaps_text)
                (out / f"{args.kind}-smaps-rollup-{safe}.txt").write_text(rollup_text)
                (out / f"{args.kind}-status-{safe}.txt").write_text(status_text)
                (out / f"{args.kind}-maps-{safe}.txt").write_text(maps_text)
                rollup = parse_rollup(rollup_text)
                status = parse_status(status_text)
                phase_rows.append({"phase": event, "trace_timestamp_us": timestamp, "pid": event_pid, "tid": tid, "detail": detail, **rollup, **{f"status_{k}": v for k, v in status.items() if k in ("VmRSS", "VmSize", "RssAnon", "RssFile", "VmSwap")}})
                for mapping in parse_smaps(smaps_text):
                    if mapping["size_bytes"] < 16 * 1024 * 1024:
                        continue
                    mapping["phase"] = event
                    mapping["kind"] = args.kind
                    mapping_rows.append(mapping)
                    pages, error = resident_pages(mapping["start"], mapping["size_bytes"])
                    residency_rows.append({"phase": event, "kind": args.kind, "start": hex(mapping["start"]), "end": hex(mapping["end"]), "size_bytes": mapping["size_bytes"], "rss_bytes": mapping.get("Rss", 0), "pss_bytes": mapping.get("Pss", 0), "path": mapping["path"], "resident_pages": pages if pages is not None else "", "resident_bytes": pages * PAGE if pages is not None else "", "mincore_status": error})
            time.sleep(0.05)
        return_code = process.wait()

    write_csv(out / "phase-memory.csv", phase_rows, sorted({key for row in phase_rows for key in row}))
    write_csv(out / "mappings.csv", mapping_rows, sorted({key for row in mapping_rows for key in row}))
    write_csv(out / "residency.csv", residency_rows, sorted({key for row in residency_rows for key in row}))
    (out / "run.json").write_text(json.dumps({"kind": args.kind, "model": args.model, "return_code": return_code, "events": sorted(seen)}, indent=2) + "\n")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
