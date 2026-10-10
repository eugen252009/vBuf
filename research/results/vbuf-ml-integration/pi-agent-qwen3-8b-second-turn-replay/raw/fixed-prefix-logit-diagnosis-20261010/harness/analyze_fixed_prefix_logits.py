#!/usr/bin/env python3
"""Offline analysis of captured Qwen3 fixed-prefix logits; no inference performed."""
import csv
import json
import pathlib
import sys

import numpy as np

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {sys.argv[0]} EVIDENCE_DIR")
root = pathlib.Path(sys.argv[1])
vocab = 151936
positions = [int(x) for x in (root / "positions.csv").read_text().strip().split(",")]
expected = np.fromstring(
    (root / "generated-token-ids.csv").read_text().splitlines()[0], sep=",", dtype=np.uint32
)


def load(path):
    values = np.fromfile(path, dtype="<f4").astype(np.float64)
    if values.size != vocab:
        raise ValueError(f"{path}: expected {vocab} f32 values, got {values.size}")
    return values


def top_ids(values, count=10):
    ix = np.argpartition(values, -count)[-count:]
    return ix[np.argsort(values[ix])[::-1]]


def probabilities(values):
    exp = np.exp(values - values.max())
    return exp / exp.sum()


def compare(a, b, target):
    diff = a - b
    ra = np.sqrt(np.mean(a * a))
    rb = np.sqrt(np.mean(b * b))
    ta, tb = top_ids(a), top_ids(b)
    pa, pb = probabilities(a), probabilities(b)
    return {
        "relative_rms": float(np.sqrt(np.mean(diff * diff)) / rb),
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "centered_cosine": float(
            np.dot(a - a.mean(), b - b.mean())
            / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()))
        ),
        "max_abs_difference": float(np.max(np.abs(diff))),
        "top1_a": int(ta[0]),
        "top1_b": int(tb[0]),
        "top1_match": bool(ta[0] == tb[0]),
        "top1_margin_a": float(a[ta[0]] - a[ta[1]]),
        "top1_margin_b": float(b[tb[0]] - b[tb[1]]),
        "target_id": int(target),
        "target_rank_a": int(1 + np.count_nonzero(a > a[target])),
        "target_rank_b": int(1 + np.count_nonzero(b > b[target])),
        "target_logit_a": float(a[target]),
        "target_logit_b": float(b[target]),
        "target_probability_a": float(pa[target]),
        "target_probability_b": float(pb[target]),
        "top10_a": [int(x) for x in ta],
        "top10_b": [int(x) for x in tb],
        "top10_intersection": int(len(set(map(int, ta)) & set(map(int, tb)))),
        "softmax_total_variation": float(0.5 * np.abs(pa - pb).sum()),
    }


def vbuf_path(kind, pos):
    if kind == "incremental":
        return root / "vbuf" / "incremental" / f"vbuf-{pos}.f32"
    return root / "vbuf" / f"fresh-target-{pos}" / "vbuf-0.f32"


comparisons = []
for pos in positions:
    v_incremental = load(vbuf_path("incremental", pos))
    v_fresh = load(vbuf_path("fresh", pos))
    llama = load(root / "llama" / f"llama-target-{pos}.f32")
    target = int(expected[pos])
    comparisons.append(
        {
            "target_index": pos,
            "input_prefix_tokens": 916 + pos,
            "expected_token": target,
            "vbuf_incremental_vs_llama": compare(v_incremental, llama, target),
            "vbuf_fresh_vs_llama": compare(v_fresh, llama, target),
            "vbuf_fresh_vs_incremental": compare(v_fresh, v_incremental, target),
        }
    )

# The first position after which the complete generated suffix is period-3.
tokens = [int(x) for x in expected]
replay_prefix_checks = {}
for count in (63, 1350):
    replay = [int(x) for x in (root / "vbuf" / f"incremental-generated-token-ids-{count}.csv")
              .read_text().strip().split(",") if x]
    replay_prefix_checks[str(count)] = {
        "observed_tokens": len(replay),
        "matches_saved_prefix": replay == tokens[:count],
    }
    if replay != tokens[:count]:
        raise ValueError(f"vBuf {count}-token replay does not match saved output prefix")
last_nonperiodic = max(
    (i for i in range(len(tokens) - 3) if tokens[i] != tokens[i + 3]), default=-1
)
period_start = last_nonperiodic + 1

with (root / "llama" / "greedy-fixed-prefix-top1.csv").open(newline="") as stream:
    trajectory_rows = list(csv.DictReader(stream))
trajectory_mismatches = [
    {
        "target_index": int(row["target_index"]),
        "vbuf_token": int(row["expected_token"]),
        "llama_top1": int(row["llama_top1"]),
        "llama_target_rank": int(row["target_rank"]),
        "llama_top1_margin": float(row["top1_margin"]),
    }
    for row in trajectory_rows
    if int(row["expected_token"]) != int(row["llama_top1"])
]

report = {
    "schema": "qwen3-fixed-prefix-logit-diagnosis-v1",
    "model_identity": {
        "gguf_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        "vbuf_payload_sha256": "cc85fa7afd90808484485de0e0a09e88ff5b98b58d1fb69c7083f64916417cb5",
        "semantic_sidecar_sha256": "99f2892dbe58d427605457729a6edcfa037d67f8ec1ff7f325953345b5d11212",
        "llama_cpp_commit": "a97123e497968f3440264c0464a7adc7c999c027",
    },
    "prefix": {
        "rendered_prompt_sha256": "39c32c329cd2683ed946e1903282ffe3afb5f54c4179de436cd15dcda018e99c",
        "prompt_tokens": 916,
        "token_id_sha256": "feaf88943eafee31328b6ed0c9a3fc7db5ceb82b393f8a6bde5d9d5637be2236",
    },
    "saved_generation_tokens": len(tokens),
    "vbuf_replay_prefix_checks": replay_prefix_checks,
    "repetition": {
        "index_base": 0,
        "period": 3,
        "first_periodic_suffix_index": period_start,
        "pattern_token_ids": tokens[period_start : period_start + 3],
        "suffix_token_count": len(tokens) - period_start,
        "complete_cycles": (len(tokens) - period_start) // 3,
        "partial_cycle_tokens": (len(tokens) - period_start) % 3,
    },
    "fixed_prefix_greedy_alignment": {
        "targets_tested": len(trajectory_rows),
        "first_mismatch_index": trajectory_mismatches[0]["target_index"] if trajectory_mismatches else None,
        "mismatch_count": len(trajectory_mismatches),
        "mismatches": trajectory_mismatches,
        "conditioning_note": "Each target is evaluated after the saved vBuf-generated tokens before that target; after the first mismatch, this is not the llama model's own trajectory.",
    },
    "logit_vocabulary": vocab,
    "captured_target_indices": positions,
    "comparisons": comparisons,
}
(root / "logit-comparison.json").write_text(json.dumps(report, indent=2) + "\n")

with (root / "logit-metrics.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["target_index", "expected", "backend_pair", "relative_rms", "cosine", "softmax_tv",
         "top1_a", "top1_b", "margin_a", "margin_b", "target_rank_a", "target_rank_b"]
    )
    for row in comparisons:
        for pair in ("vbuf_incremental_vs_llama", "vbuf_fresh_vs_llama", "vbuf_fresh_vs_incremental"):
            item = row[pair]
            writer.writerow(
                [row["target_index"], row["expected_token"], pair, item["relative_rms"],
                 item["cosine"], item["softmax_total_variation"], item["top1_a"], item["top1_b"],
                 item["top1_margin_a"], item["top1_margin_b"], item["target_rank_a"], item["target_rank_b"]]
            )

print(json.dumps({
    "period3_start": period_start,
    "period3_pattern": tokens[period_start : period_start + 3],
    "period3_suffix_tokens": len(tokens) - period_start,
    "llama_fixed_prefix_targets": len(trajectory_rows),
    "first_llama_top1_mismatch": trajectory_mismatches[0] if trajectory_mismatches else None,
    "llama_top1_mismatches": trajectory_mismatches,
    "analysis": str(root / "logit-comparison.json"),
}, indent=2))
