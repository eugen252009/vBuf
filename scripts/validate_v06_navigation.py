#!/usr/bin/env python3
"""Validate Step 5A raw benchmark shape and emit measured summaries."""
from __future__ import annotations
import csv, os, platform, statistics, sys
from collections import defaultdict

BASE_STEPS = {8, 16, 32, 64, 128, 256}
LAYOUTS = {"many-tiny", "few-large", "mixed", "aos", "soa", "composite-continuation", "large-opaque", "zero-and-partial"}
REQUIRED_VARIANTS = {
    ("canonical-validation", "parse-and-describe"),
    ("nano-construction", "reconstruct-from-validated-canonical"),
    ("nano-deployment", "embedded-load"),
    ("nano-deployment", "reconstructed-cache-load"),
    ("canonical-block-traversal", "canonical"),
    ("physical-start-enumeration", "nano"),
    ("nth-start", "nano+checkpoint-512"),
    ("nth-start", "nano+checkpoint-4096"),
    ("nth-start", "nano+checkpoint-65536"),
    ("key-lookup", "canonical-linear-scan"),
    ("key-lookup", "directory-binary-search"),
    ("parallel-payload-sum", "canonical-rayon"),
    ("parallel-start-enumeration", "nano-rayon"),
}

def median(rows):
    return statistics.median(int(row["nanos"]) for row in rows)

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_v06_navigation.py RAW.csv")
    raw = sys.argv[1]
    with open(raw, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("raw report is empty")
    bases = {int(row["base_step"]) for row in rows}
    layouts = {row["layout"] for row in rows}
    if bases != BASE_STEPS:
        raise SystemExit(f"BaseStep set mismatch: {sorted(bases)}")
    if layouts != LAYOUTS:
        raise SystemExit(f"layout set mismatch: {sorted(layouts)}")
    groups = defaultdict(list)
    for row in rows:
        row["base_step"] = int(row["base_step"])
        row["sample"] = int(row["sample"])
        row["nanos"] = int(row["nanos"])
        row["file_bytes"] = int(row["file_bytes"])
        row["payload_bytes"] = int(row["payload_bytes"])
        row["block_count"] = int(row["block_count"])
        row["nano_bytes"] = int(row["nano_bytes"])
        row["checkpoint_bytes"] = int(row["checkpoint_bytes"])
        row["directory_bytes"] = int(row["directory_bytes"])
        groups[(row["layout"], row["base_step"], row["operation"], row["variant"])].append(row)
    observed_variants = {(key[2], key[3]) for key in groups}
    if observed_variants != REQUIRED_VARIANTS:
        raise SystemExit(f"operation/variant mismatch: {sorted(observed_variants ^ REQUIRED_VARIANTS)}")
    expected_groups = len(LAYOUTS) * len(BASE_STEPS) * len(REQUIRED_VARIANTS)
    if len(groups) != expected_groups:
        raise SystemExit(f"expected {expected_groups} groups, found {len(groups)}")
    for key, group in groups.items():
        if len(group) != 20 or {row["sample"] for row in group} != set(range(20)):
            raise SystemExit(f"sample shape mismatch for {key}")
        if any(row["nanos"] < 0 for row in group):
            raise SystemExit(f"negative duration for {key}")
    for layout in LAYOUTS:
        for base in BASE_STEPS:
            metadata = [row for key, group in groups.items() if key[:2] == (layout, base) for row in group[:1]]
            payload = {row["payload_bytes"] for row in metadata}
            blocks = {row["block_count"] for row in metadata}
            if len(payload) != 1 or len(blocks) != 1:
                raise SystemExit(f"metadata is not stable for {layout}/{base}")
    report = os.path.splitext(raw)[0] + ".md"
    with open(report, "w") as out:
        out.write("# Step 5A raw navigation qualification\n\n")
        out.write("This report is generated from the CSV; timings are measurements, not wire decisions.\n\n")
        out.write(f"- CPU: `{platform.processor() or 'unreported'}`\n- Python validator: `{platform.python_version()}`\n")
        out.write(f"- Raw rows: {len(rows)}; samples/group: 20; BaseSteps: {', '.join(map(str, sorted(BASE_STEPS)))} bytes\n\n")
        out.write("## Median nanoseconds by operation (all layouts pooled per BaseStep)\n\n")
        out.write("| BaseStep | Operation | Variant | Median ns |\n|---:|---|---|---:|\n")
        pooled = defaultdict(list)
        for (layout, base, operation, variant), group in groups.items():
            pooled[(base, operation, variant)].extend(group)
        for (base, operation, variant), group in sorted(pooled.items()):
            out.write(f"| {base} | {operation} | {variant} | {median(group):,} |\n")
        out.write("\n## Measured facts and conservative decisions\n\n")
        out.write("- **Measured fact:** all six frozen BaseSteps and all eight generic layouts completed with stable metadata and 20 samples per operation.\n")
        out.write("- **Measured fact:** Nano bytes and padding vary with BaseStep and topology; see the raw `file_bytes`, `payload_bytes`, `nano_bytes`, and checkpoint/directory columns.\n")
        out.write("- **Inference:** these in-memory timings do not establish a universal BaseStep winner, nor do they include OS-controlled cold-cache state.\n")
        out.write("- **Decision:** BaseStep remains a legal generic tuning input; no BaseStep is promoted or removed by this run.\n")
        out.write("- **Decision:** Nano, checkpoints, and region-directory paths remain `POSSIBLE BUT NOT YET JUSTIFIED` pending independent end-to-end qualification including construction, persistence, validation, and equivalent lookup work.\n")
        out.write("- **Decision:** no experimental artifact becomes normative and no finalization envelope is added in Step 5A.\n")
        out.write("\n## Environment limitations\n\n")
        out.write("The Rust runner records compiler/build context in the repository; this validation report intentionally does not infer cache, fault, SIMD, or pointer-alignment claims that were not instrumented.\n")
    print(f"validated {len(rows)} raw rows across {len(groups)} groups; wrote {report}")

if __name__ == "__main__":
    main()
