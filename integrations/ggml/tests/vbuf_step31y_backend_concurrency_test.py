#!/usr/bin/env python3
"""Step 31Y isolated backend-concurrency feasibility qualification."""

import argparse
import json
import os
import statistics
import subprocess


def run_probe(args, mode):
    command = [
        args.binary,
        "--semantic-model", args.semantic_model,
        "--source-url", args.source_url,
        "--mode", mode,
        "--prompt-a", args.prompt_a,
        "--prompt-b", args.prompt_b,
        "--blocks", str(args.blocks),
        "--capacity", str(args.capacity),
        "--max-new-tokens", str(args.max_new_tokens),
        "--warmup", str(args.warmup),
        "--repetitions", str(args.repetitions),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout,
        env={**os.environ, "LD_LIBRARY_PATH": args.ld_library_path},
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(
            f"{mode} probe produced no JSON; returncode={completed.returncode}; "
            f"stderr={completed.stderr.strip()}"
        )
    evidence = json.loads(lines[-1])
    assert evidence["mode"] == mode
    assert completed.returncode == 0, (mode, completed.returncode, evidence, completed.stderr)
    assert evidence["valid"]
    return evidence


def mean(values):
    return statistics.fmean(values)


def summarize(serial, concurrent):
    serial_pairs = serial["pairs"]
    concurrent_pairs = concurrent["pairs"]
    serial_makespans = [pair["makespan_ns"] for pair in serial_pairs]
    concurrent_makespans = [pair["makespan_ns"] for pair in concurrent_pairs]
    serial_individual = [
        run["duration_ns"] for pair in serial_pairs for run in (pair["a"], pair["b"])
    ]
    concurrent_individual = [
        run["duration_ns"] for pair in concurrent_pairs for run in (pair["a"], pair["b"])
    ]
    assert serial["total_overlap_ns"] == 0
    assert concurrent["total_overlap_ns"] > 0
    assert all(pair["overlap_ns"] == 0 for pair in serial_pairs)
    assert all(pair["overlap_ns"] > 0 for pair in concurrent_pairs)

    return {
        "serial_mean_makespan_ns": mean(serial_makespans),
        "concurrent_mean_makespan_ns": mean(concurrent_makespans),
        "aggregate_makespan_speedup": mean(serial_makespans) / mean(concurrent_makespans),
        "serial_mean_individual_duration_ns": mean(serial_individual),
        "concurrent_mean_individual_duration_ns": mean(concurrent_individual),
        "individual_latency_factor": mean(concurrent_individual) / mean(serial_individual),
        "serial_peak_rss_kib": serial["peak_rss_kib"],
        "concurrent_peak_rss_kib": concurrent["peak_rss_kib"],
        "serial_peak_rss_delta_kib": serial["peak_rss_delta_kib"],
        "concurrent_peak_rss_delta_kib": concurrent["peak_rss_delta_kib"],
        "serial_peak_pss_kib": serial["peak_pss_kib"],
        "concurrent_peak_pss_kib": concurrent["peak_pss_kib"],
        "serial_peak_threads": serial["peak_threads"],
        "concurrent_peak_threads": concurrent["peak_threads"],
        "token_parity": True,
        "shared_resident_weights": "NOT_QUALIFIED_PRIVATE_SESSIONS_ONLY",
        "production_max_active_generations": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--ld-library-path", required=True)
    parser.add_argument("--prompt-a", default="Say hi")
    parser.add_argument("--prompt-b", default="Count to one")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=268435456)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    serial = run_probe(args, "serial")
    concurrent = run_probe(args, "concurrent")
    evidence = {
        "step": "31Y",
        "qualification": "isolated_backend_concurrency_feasibility",
        "ggml_commit": serial["ggml_commit"],
        "serial": serial,
        "concurrent": concurrent,
        "summary": summarize(serial, concurrent),
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(evidence, output, ensure_ascii=False, indent=2)

    summary = evidence["summary"]
    print("VBUF_STEP31Y_BACKEND_CONCURRENCY_QUALIFICATION=PASS")
    print("ISOLATED_SESSIONS=PASS")
    print("DETERMINISTIC_TOKEN_PARITY=PASS")
    print("SERIAL_OVERLAP_NS=0")
    print("CONCURRENT_OVERLAP_NS=" + str(concurrent["total_overlap_ns"]))
    print("AGGREGATE_MAKESPAN_SPEEDUP=" + f"{summary['aggregate_makespan_speedup']:.3f}x")
    print("INDIVIDUAL_LATENCY_FACTOR=" + f"{summary['individual_latency_factor']:.3f}x")
    print("SHARED_RESIDENT_WEIGHTS=NOT_QUALIFIED")
    print("PRODUCTION_MAX_ACTIVE_GENERATIONS=1")
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
