#!/usr/bin/env python3
"""Qualify Qwen3 GPT-2 BPE/Qwen2 tokenizer data against pinned llama.cpp."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import EXPECTED, Reader, parse  # noqa: E402
from qualify_step17 import PINNED_COMMIT  # noqa: E402

TOKENIZER_KEYS = (
    "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token", "tokenizer.chat_template",
)

SOURCE = {
    "src/llama-vocab.cpp": [
        "void llama_vocab::impl::load",
        "ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model)",
        "tokenizer_model == \"gpt2\"",
        "const size_t pos = word.find(' ', 1)",
        "tokenizer_pre == \"qwen2\"",
        "gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST)",
        "gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE)",
        "LLM_KV_TOKENIZER_ADD_BOS",
        "special_token_types",
    ],
    "src/llama-model.cpp": [
        "llama_model_chat_template",
        "LLM_KV_TOKENIZER_CHAT_TEMPLATE",
    ],
}


def type_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("rb") as stream:
        reader = Reader(stream)
        if reader.read(4) != b"GGUF":
            raise ValueError("invalid GGUF magic")
        reader.read(4)
        metadata_count = int(reader.scalar("Q"))
        tensor_count = int(reader.scalar("Q"))
        # GGUF header order is tensor_count then metadata_count; correct the
        # local variables after reading the two u64 values.
        metadata_count, tensor_count = tensor_count, metadata_count
        for _ in range(metadata_count):
            key = reader.string()
            value_type = int(reader.scalar("I"))
            if value_type == 8:
                result[key] = "string"
                reader.string()
            elif value_type == 9:
                element_type = int(reader.scalar("I"))
                count = int(reader.scalar("Q"))
                result[key] = f"array[{element_type}]"
                for _ in range(count):
                    reader.value(element_type)
            else:
                names = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool", 10: "u64", 11: "i64", 12: "f64"}
                result[key] = names.get(value_type, f"type-{value_type}")
                reader.value(value_type)
    return result


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sequence_digest(values: list[object]) -> str:
    encoded = bytearray()
    for value in values:
        raw = value.encode() if isinstance(value, str) else struct.pack("<q", int(value))
        encoded.extend(struct.pack("<Q", len(raw)))
        encoded.extend(raw)
    return digest_bytes(bytes(encoded))


def source_evidence(root: Path) -> dict[str, list[dict[str, object]]]:
    evidence = {}
    for relative, needles in SOURCE.items():
        lines = (root / relative).read_text(encoding="utf-8").splitlines()
        matches = []
        for needle in needles:
            match = next(((i + 1, line.strip()) for i, line in enumerate(lines) if needle in line), None)
            if match is None:
                raise ValueError(f"missing pinned source evidence: {relative}: {needle}")
            matches.append({"needle": needle, "line": match[0], "text": match[1]})
        evidence[relative] = matches
    return evidence


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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
    output = (args.output_dir or root / "benchmark-results" / "vbuf-ml-step19b").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not upstream.exists():
        print(f"SKIP — pinned upstream checkout absent: {upstream}")
        return 0
    try:
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        if commit != PINNED_COMMIT:
            print(f"FAIL — upstream commit mismatch: expected {PINNED_COMMIT}, got {commit}", file=sys.stderr)
            return 1
        provenance = source_evidence(upstream)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"FAIL — pinned tokenizer provenance: {error}", file=sys.stderr)
        return 1

    artifacts = {}
    inventories = {}
    for label, (filename, expected_hash) in EXPECTED.items():
        path = root / "research-models" / filename
        if not path.exists():
            print(f"SKIP — research model absent: {path}")
            return 0
        artifact = parse(path)
        if artifact.sha256 != expected_hash:
            print(f"FAIL — hash mismatch for {path}", file=sys.stderr)
            return 1
        artifacts[label] = artifact
        inventories[label] = type_inventory(path)
        missing = [key for key in TOKENIZER_KEYS if key not in artifact.metadata]
        if missing:
            print(f"FAIL — {label} missing tokenizer keys: {missing}", file=sys.stderr)
            return 1

    common = artifacts["Q8_0"].metadata
    for label, artifact in artifacts.items():
        if artifact.metadata[TOKENIZER_KEYS[0]] != "gpt2" or artifact.metadata[TOKENIZER_KEYS[1]] != "qwen2":
            print(f"FAIL — {label} is not the qualified gpt2/qwen2 pair", file=sys.stderr)
            return 1
        if artifact.metadata["tokenizer.ggml.add_bos_token"] is not False:
            print(f"FAIL — {label} add_bos is not false", file=sys.stderr)
            return 1
        if len(artifact.metadata["tokenizer.ggml.tokens"]) != len(artifact.metadata["tokenizer.ggml.token_type"]):
            print(f"FAIL — {label} vocabulary/type lengths differ", file=sys.stderr)
            return 1
        tokens = artifact.metadata["tokenizer.ggml.tokens"]
        token_ids = {token: index for index, token in enumerate(tokens)}
        merges = artifact.metadata["tokenizer.ggml.merges"]
        left_ids: list[int] = []
        right_ids: list[int] = []
        malformed = 0
        missing = 0
        for merge in merges:
            pos = merge.find(" ", 1)
            if pos < 0:
                malformed += 1
                continue
            left, right = merge[:pos], merge[pos + 1:]
            if left not in token_ids or right not in token_ids:
                missing += 1
                continue
            left_ids.append(token_ids[left]); right_ids.append(token_ids[right])
        if malformed or missing or len(left_ids) != len(merges):
            print(f"FAIL — {label} merge resolution malformed={malformed} missing={missing}", file=sys.stderr)
            return 1
        if len(set(zip(left_ids, right_ids))) != len(merges):
            print(f"FAIL — {label} contains duplicate merge pairs", file=sys.stderr)
            return 1
        artifacts[label].resolved_left = left_ids  # type: ignore[attr-defined]
        artifacts[label].resolved_right = right_ids  # type: ignore[attr-defined]

    inventory_rows = []
    for label, artifact in artifacts.items():
        for key in TOKENIZER_KEYS:
            value = artifact.metadata[key]
            if isinstance(value, list):
                shape = f"array[{len(value)}]"
                size = sum(len(v.encode("utf-8")) if isinstance(v, str) else 4 for v in value)
            elif isinstance(value, str):
                shape = f"string[{len(value.encode('utf-8'))} bytes]"
                size = len(value.encode("utf-8"))
            else:
                shape = repr(value)
                size = 1 if isinstance(value, bool) else 4
            inventory_rows.append({"artifact": label, "key": key, "gguf_type": inventories[label].get(key, "unknown"),
                "value_count_or_size": shape, "semantic": {"tokenizer.ggml.model": "algorithm identity",
                "tokenizer.ggml.pre": "pre-tokenizer identity", "tokenizer.ggml.tokens": "vocabulary ordinal text",
                "tokenizer.ggml.token_type": "consumer token attributes", "tokenizer.ggml.merges": "ranked BPE source pairs",
                "tokenizer.ggml.bos_token_id": "BOS ID", "tokenizer.ggml.eos_token_id": "EOS ID",
                "tokenizer.ggml.padding_token_id": "PAD ID", "tokenizer.ggml.add_bos_token": "BOS insertion policy",
                "tokenizer.chat_template": "opaque chat-template source"}[key], "payload_bytes_estimate": size,
                "same_across_artifacts": value == common[key]})
    write_csv(output / "tokenizer-source-matrix.csv", inventory_rows)

    merge_rows = []
    for label, artifact in artifacts.items():
        merges = artifact.metadata["tokenizer.ggml.merges"]
        source_bytes = sum(len(value.encode("utf-8")) for value in merges) + (len(merges) + 1) * 8
        numeric_bytes = len(merges) * 4 * 2
        merge_rows.append({"artifact": label, "merge_count": len(merges), "source_string_utf8_plus_u64_offsets": source_bytes,
            "numeric_pair_u32_arrays": numeric_bytes, "numeric_savings_bytes": source_bytes - numeric_bytes,
            "source_split_rule": "first ASCII space at or after byte index 1",
            "selected": "numeric pair arrays; rank is array ordinal", "resolution": "all pairs resolve against vocabulary ordinals",
            "duplicate_pairs": 0, "runtime_work": "resolve IDs to vocabulary text while constructing pinned bpe_ranks"})
    write_csv(output / "merge-representation-comparison.csv", merge_rows)

    parity_rows = []
    checks = [
        ("vocabulary", lambda a: (len(a.metadata["tokenizer.ggml.tokens"]), sequence_digest(a.metadata["tokenizer.ggml.tokens"]))),
        ("token_types", lambda a: (len(a.metadata["tokenizer.ggml.token_type"]), sequence_digest(a.metadata["tokenizer.ggml.token_type"]))),
        ("merges", lambda a: (len(a.metadata["tokenizer.ggml.merges"]), sequence_digest(a.metadata["tokenizer.ggml.merges"]))),
        ("resolved_merge_left_ids", lambda a: (len(a.resolved_left), sequence_digest(a.resolved_left))),  # type: ignore[attr-defined]
        ("resolved_merge_right_ids", lambda a: (len(a.resolved_right), sequence_digest(a.resolved_right))),  # type: ignore[attr-defined]
        ("model_identity", lambda a: (1, a.metadata["tokenizer.ggml.model"])),
        ("pre_tokenizer_identity", lambda a: (1, a.metadata["tokenizer.ggml.pre"])),
        ("bos_id", lambda a: (1, a.metadata["tokenizer.ggml.bos_token_id"])),
        ("eos_id", lambda a: (1, a.metadata["tokenizer.ggml.eos_token_id"])),
        ("pad_id", lambda a: (1, a.metadata["tokenizer.ggml.padding_token_id"])),
        ("add_bos", lambda a: (1, a.metadata["tokenizer.ggml.add_bos_token"])),
        ("chat_template", lambda a: (len(a.metadata["tokenizer.chat_template"].encode()), digest_bytes(a.metadata["tokenizer.chat_template"].encode()))),
    ]
    for name, getter in checks:
        q8 = getter(artifacts["Q8_0"]); bf16 = getter(artifacts["BF16"])
        parity_rows.append({"semantic": name, "q8_0_count_or_size": q8[0], "bf16_count_or_size": bf16[0],
            "q8_0_digest_or_value": q8[1], "bf16_digest_or_value": bf16[1], "match": q8 == bf16,
            "classification": "DIRECTLY_REPRESENTED"})
    write_csv(output / "tokenizer-parity.csv", parity_rows)
    (output / "upstream-provenance.json").write_text(json.dumps({"repository": "https://github.com/ggml-org/llama.cpp.git", "commit": PINNED_COMMIT,
        "source": provenance, "decisions": {"model": "gpt2 -> GPT2_BPE", "pre": "qwen2 -> QWEN2", "merges": "resolved u32 vocabulary-ordinal pairs; array ordinal is rank",
        "duplicate_pair_behavior": "pinned emplace keeps first rank; real artifacts contain none", "chat_template": "opaque UTF-8 optional payload; execution remains consumer-local"}}, indent=2) + "\n", encoding="utf-8")
    (output / "qualification-config.json").write_text(json.dumps({"consumer_commit": PINNED_COMMIT, "python": sys.version, "platform": platform.platform(),
        "artifacts": {label: {"filename": artifact.path.name, "sha256": artifact.sha256, "token_count": len(artifact.metadata["tokenizer.ggml.tokens"]),
        "merge_count": len(artifact.metadata["tokenizer.ggml.merges"])} for label, artifact in artifacts.items()}}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS — Qwen3 tokenizer semantics qualified for {len(artifacts)} artifacts; evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
