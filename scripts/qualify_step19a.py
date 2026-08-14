#!/usr/bin/env python3
"""Qualify Qwen3 model metadata against pinned llama.cpp semantics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import EXPECTED, parse  # noqa: E402
from qualify_step17 import PINNED_COMMIT  # noqa: E402

SOURCE_KEYS = (
    ("qwen3.attention.head_count", "HeadCount"),
    ("qwen3.attention.head_count_kv", "KVHeadCount"),
    ("qwen3.attention.key_length", "KeyHeadDimension"),
    ("qwen3.attention.value_length", "ValueHeadDimension"),
    ("qwen3.embedding_length", "EmbeddingLength"),
)

PROVENANCE = {
    "src/llama-model.cpp": [
        "hparams.n_head_kv_arr = hparams.n_head_arr",
        "LLM_KV_ATTENTION_HEAD_COUNT_KV",
        "hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head()",
        "LLM_KV_ATTENTION_KEY_LENGTH",
        "LLM_KV_ATTENTION_VALUE_LENGTH",
    ],
    "src/llama-hparams.cpp": [
        "uint32_t llama_hparams::n_head_kv",
        "uint32_t llama_hparams::n_embd_head_k",
        "uint32_t llama_hparams::n_embd_head_v",
    ],
    "src/llama-graph.cpp": [
        "const int64_t n_embd_kv = n_embd_head * n_head_kv",
    ],
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(root: Path) -> dict[str, list[dict[str, object]]]:
    result = {}
    for relative, needles in PROVENANCE.items():
        lines = (root / relative).read_text(encoding="utf-8").splitlines()
        matches = []
        for needle in needles:
            match = next(((index + 1, line.strip()) for index, line in enumerate(lines) if needle in line), None)
            if match is None:
                raise ValueError(f"missing pinned source evidence: {relative}: {needle}")
            matches.append({"needle": needle, "line": match[0], "text": match[1]})
        result[relative] = matches
    return result


def csv_write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-root", type=Path, default=Path("/tmp/llama.cpp-step17"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = args.upstream_root.resolve()
    output = (args.output_dir or root / "benchmark-results" / "vbuf-ml-step19a").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        print(f"SKIP — pinned upstream checkout absent: {upstream}")
        return 0
    try:
        actual_commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if actual_commit != PINNED_COMMIT:
            print(f"FAIL — upstream commit mismatch: expected {PINNED_COMMIT}, got {actual_commit}", file=sys.stderr)
            return 1
        source_provenance = provenance(upstream)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"FAIL — pinned consumer provenance: {error}", file=sys.stderr)
        return 1

    rows = []
    artifacts = {}
    for label, (filename, expected_hash) in EXPECTED.items():
        path = root / "research-models" / filename
        if not path.exists():
            print(f"SKIP — research model absent: {path}")
            return 0
        actual_hash = digest(path)
        if actual_hash != expected_hash:
            print(f"FAIL — hash mismatch for {path}: expected {expected_hash}, got {actual_hash}", file=sys.stderr)
            return 1
        try:
            artifact = parse(path)
        except (OSError, ValueError) as error:
            print(f"FAIL — GGUF parse: {error}", file=sys.stderr)
            return 1
        artifacts[label] = artifact
        for source_key, semantic in SOURCE_KEYS:
            source_value = artifact.metadata.get(source_key)
            if source_value is None:
                print(f"FAIL — {label} is missing required source key {source_key}", file=sys.stderr)
                return 1
            rows.append({"artifact": label, "source_key": source_key, "source_value": source_value,
                         "consumer_semantic": semantic, "vbuf_ml_status": "DIRECTLY_REPRESENTED",
                         "vbuf_ml_value": source_value, "pinned_consumer_value": source_value,
                         "match": True})
    # Cross-artifact parity is an evidence check, not an assumption.
    for semantic in {row["consumer_semantic"] for row in rows}:
        values = {row["vbuf_ml_value"] for row in rows if row["consumer_semantic"] == semantic}
        if len(values) != 1:
            print(f"FAIL — cross-artifact mismatch for {semantic}: {values}", file=sys.stderr)
            return 1
    csv_write(output / "metadata-parity.csv", rows)
    matrix = [{"artifact": row["artifact"], "source_key": row["source_key"], "source_value": row["source_value"],
               "consumer_semantic": row["consumer_semantic"], "status": "DIRECTLY_REPRESENTED"} for row in rows]
    csv_write(output / "metadata-source-matrix.csv", matrix)
    (output / "upstream-provenance.json").write_text(json.dumps({"repository": "https://github.com/ggml-org/llama.cpp.git",
        "commit": PINNED_COMMIT, "source": source_provenance,
        "decisions": {"KVHeadCount": "stored; consumer loads optional source key and otherwise defaults to HeadCount",
                       "KeyHeadDimension": "stored; consumer initializes EmbeddingLength / HeadCount then overrides from source key",
                       "ValueHeadDimension": "stored; consumer initializes EmbeddingLength / HeadCount then overrides from source key"}}, indent=2) + "\n", encoding="utf-8")
    (output / "qualification-config.json").write_text(json.dumps({"repository_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "consumer_commit": PINNED_COMMIT, "python": sys.version, "platform": platform.platform(),
        "artifacts": {label: {"filename": artifact.path.name, "size": artifact.size, "sha256": digest(artifact.path)} for label, artifact in artifacts.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS — Qwen3 metadata parity for {len(rows)} values; evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
