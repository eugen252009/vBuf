#!/usr/bin/env python3
"""Step 31Z shared-residency and concurrent immutable-substrate qualification."""

import argparse
import json
import os
import statistics
import subprocess


def run_probe(args, mode):
    completed = subprocess.run([
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
    ], check=False, capture_output=True, text=True, timeout=args.timeout,
        env={**os.environ, "LD_LIBRARY_PATH": args.ld_library_path})
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"{mode} produced no evidence: {completed.stderr.strip()}"
    evidence = json.loads(lines[-1])
    assert completed.returncode == 0, (mode, completed.returncode, evidence, completed.stderr)
    assert evidence["mode"] == mode and evidence["valid"]
    return evidence


def mean(values):
    return statistics.fmean(values)


def validate_modes(serial, isolated, shared_serial, shared_concurrent):
    assert serial["total_overlap_ns"] == 0
    assert isolated["total_overlap_ns"] > 0
    assert shared_serial["total_overlap_ns"] == 0
    assert shared_concurrent["total_overlap_ns"] > 0
    assert all(pair["overlap_ns"] == 0 for pair in serial["pairs"])
    assert all(pair["overlap_ns"] > 0 for pair in isolated["pairs"])
    assert all(pair["overlap_ns"] == 0 for pair in shared_serial["pairs"])
    assert all(pair["overlap_ns"] > 0 for pair in shared_concurrent["pairs"])

    expected = [
        [(pair["a"]["token_hash"], pair["b"]["token_hash"]) for pair in mode["pairs"]]
        for mode in (serial, isolated, shared_serial, shared_concurrent)
    ]
    assert all(tokens == expected[0] for tokens in expected[1:])


def summarize(serial, isolated, shared_serial, shared_concurrent):
    def makespans(evidence):
        return [pair["makespan_ns"] for pair in evidence["pairs"]]

    def individual(evidence):
        return [run["duration_ns"] for pair in evidence["pairs"] for run in (pair["a"], pair["b"])]

    serial_pairs = makespans(serial)
    isolated_pairs = makespans(isolated)
    shared_pairs = makespans(shared_concurrent)
    serial_individual = individual(serial)
    shared_individual = individual(shared_concurrent)
    return {
        "serial_pair_p50_ns": statistics.median(serial_pairs),
        "isolated_concurrent_pair_p50_ns": statistics.median(isolated_pairs),
        "shared_concurrent_pair_p50_ns": statistics.median(shared_pairs),
        "serial_mean_makespan_ns": mean(serial_pairs),
        "isolated_concurrent_mean_makespan_ns": mean(isolated_pairs),
        "shared_concurrent_mean_makespan_ns": mean(shared_pairs),
        "isolated_speedup_vs_serial": mean(serial_pairs) / mean(isolated_pairs),
        "shared_speedup_vs_serial": mean(serial_pairs) / mean(shared_pairs),
        "shared_latency_factor_vs_serial": mean(shared_individual) / mean(serial_individual),
        "serial_peak_rss_kib": serial["peak_rss_kib"],
        "isolated_concurrent_peak_rss_kib": isolated["peak_rss_kib"],
        "shared_concurrent_peak_rss_kib": shared_concurrent["peak_rss_kib"],
        "isolated_concurrent_rss_savings_kib": isolated["peak_rss_kib"] - shared_concurrent["peak_rss_kib"],
        "shared_concurrent_peak_threads": shared_concurrent["peak_threads"],
        "isolated_concurrent_peak_threads": isolated["peak_threads"],
        "residency_cap_bytes": shared_concurrent["residency_capacity"],
        "token_parity": True,
        "shared_resident_weights": "QUALIFIED_FOR_THIS_SUBSTRATE_SEAM",
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
    parser.add_argument("--capacity", type=int, default=536870912)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()

    serial = run_probe(args, "serial")
    isolated = run_probe(args, "concurrent")
    shared_serial = run_probe(args, "shared-serial")
    shared_concurrent = run_probe(args, "shared-concurrent")
    validate_modes(serial, isolated, shared_serial, shared_concurrent)
    evidence = {
        "step": "31Z",
        "qualification": "shared_residency_concurrency",
        "ggml_commit": serial["ggml_commit"],
        "serial": serial,
        "isolated_concurrent": isolated,
        "shared_serial": shared_serial,
        "shared_concurrent": shared_concurrent,
        "summary": summarize(serial, isolated, shared_serial, shared_concurrent),
    }
    with open(args.evidence, "w", encoding="utf-8") as output:
        json.dump(evidence, output, ensure_ascii=False, indent=2)

    summary = evidence["summary"]
    print("VBUF_STEP31Z_SHARED_RESIDENCY_QUALIFICATION=PASS")
    print("DUAL_READER_SUBSTRATE=PASS")
    print("SHARED_CONCURRENT_TOKEN_PARITY=PASS")
    print("ISOLATED_SPEEDUP=" + f"{summary['isolated_speedup_vs_serial']:.3f}x")
    print("SHARED_SPEEDUP=" + f"{summary['shared_speedup_vs_serial']:.3f}x")
    print("SHARED_LATENCY_FACTOR=" + f"{summary['shared_latency_factor_vs_serial']:.3f}x")
    print("SHARED_RSS_SAVINGS_KIB=" + str(summary["isolated_concurrent_rss_savings_kib"]))
    print("PRODUCTION_MAX_ACTIVE_GENERATIONS=1")
    print("EVIDENCE=" + args.evidence)


if __name__ == "__main__":
    main()
