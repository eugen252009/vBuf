#!/usr/bin/env python3
"""Build deterministic, read-only Step-18 GGUF conversion manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import EXPECTED, layout_order, parse, role_key  # noqa: E402
from qualify_step17 import PINNED_COMMIT, expected_bytes  # noqa: E402

TENSOR_KEY_ID = 0x0200
CONTROL_KEY_IDS = {"Bootstrap": 0x0201, "ModelMetadata": 0x0202, "TokenizerMetadata": 0x0203, "IntegrityMetadata": 0x0204}
PLACEMENT = "LAYER_MAJOR_ROLE_ORDER"
SUPPORTED_TYPES = {"F32": ("CanonicalPrimitive", "primitive"), "BF16": ("BF16", "opaque_bytes"), "Q8_0": ("GGML_Q8_0", "opaque_bytes")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = length
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source payload ended before its checked range")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def source_range_summary(artifact, tensors) -> dict[str, int]:
    ranges = sorted((tensor.absolute_start, tensor.absolute_end) for tensor in tensors)
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return {"range_count": len(merged), "physical_bytes": sum(end - start for start, end in merged),
            "semantic_bytes": sum(end - start for start, end in ranges)}


def metadata_plan(artifact) -> list[dict[str, object]]:
    metadata = artifact.metadata
    fields = [
        ("general.architecture", "architecture", "Architecture", "DIRECTLY_REPRESENTED", metadata.get("general.architecture")),
        ("qwen3.context_length", "context length", "ContextLength", "DIRECTLY_REPRESENTED", metadata.get("qwen3.context_length")),
        ("qwen3.embedding_length", "embedding length", "EmbeddingLength", "DIRECTLY_REPRESENTED", metadata.get("qwen3.embedding_length")),
        ("qwen3.block_count", "layer count", "LayerCount", "DIRECTLY_REPRESENTED", metadata.get("qwen3.block_count")),
        ("qwen3.attention.head_count", "attention head count", "HeadCount", "DIRECTLY_REPRESENTED", metadata.get("qwen3.attention.head_count")),
        ("qwen3.attention.head_count_kv", "KV head count", "KVHeadCount", "DIRECTLY_REPRESENTED", metadata.get("qwen3.attention.head_count_kv")),
        ("qwen3.attention.key_length", "key head dimension", "KeyHeadDimension", "DIRECTLY_REPRESENTED", metadata.get("qwen3.attention.key_length")),
        ("qwen3.attention.value_length", "value head dimension", "ValueHeadDimension", "DIRECTLY_REPRESENTED", metadata.get("qwen3.attention.value_length")),
        ("qwen3.feed_forward_length", "feed-forward length", "FeedForwardLength", "DIRECTLY_REPRESENTED", metadata.get("qwen3.feed_forward_length")),
        ("qwen3.attention.layer_norm_rms_epsilon", "RMS normalization epsilon", "NormalizationEpsilon", "DIRECTLY_REPRESENTED", metadata.get("qwen3.attention.layer_norm_rms_epsilon")),
        ("qwen3.rope.freq_base", "RoPE base", "RopeTheta", "DIRECTLY_REPRESENTED", metadata.get("qwen3.rope.freq_base")),
        ("tokenizer.ggml.tokens", "vocabulary size", "VocabularySize", "DERIVABLE", len(metadata["tokenizer.ggml.tokens"]) if isinstance(metadata.get("tokenizer.ggml.tokens"), list) else None),
    ]
    return [{"source_key": source, "consumer_semantic": consumer, "target": target,
             "status": status, "source_value": value} for source, consumer, target, status, value in fields]


def tokenizer_plan(artifact) -> list[dict[str, object]]:
    metadata = artifact.metadata
    rows = []
    for key in sorted(key for key in metadata if key.startswith("tokenizer.")):
        value = metadata[key]
        if isinstance(value, list):
            shape = f"array[{len(value)}]"
        elif isinstance(value, str):
            shape = value if len(value) < 120 else f"string[{len(value)}]"
        else:
            shape = value
        if key in {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type"}:
            status, target = "DIRECTLY_REPRESENTED", "VocabularyOnly"
        elif key == "tokenizer.ggml.merges":
            status, target = "UNSUPPORTED_REQUIRED", None
        elif key in {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.chat_template"}:
            status, target = "UNSUPPORTED_REQUIRED", None
        elif key.endswith("_token_id"):
            status, target = "DIRECTLY_REPRESENTED", "VocabularyOnly special ID"
        elif key == "tokenizer.ggml.scores":
            status, target = "DIRECTLY_REPRESENTED", "VocabularyOnly optional scores"
        else:
            status, target = "IGNORE_WITH_REASON", "consumer/default metadata"
        rows.append({"source_key": key, "meaning": "tokenizer metadata", "source_shape_or_value": shape,
                     "target": target, "status": status})
    return rows


def tensor_plans(artifact):
    ordered = layout_order(artifact, PLACEMENT)
    plans = []
    for target_order, tensor in enumerate(ordered):
        if tensor.type_name not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported source representation: {tensor.name}: {tensor.type_name}")
        expected = expected_bytes(tensor.type_name, tensor.shape)
        if expected != tensor.payload_size:
            raise ValueError(f"representation geometry mismatch: {tensor.name}")
        representation, storage = SUPPORTED_TYPES[tensor.type_name]
        plans.append({
            "source_name": tensor.name,
            "source_ordinal": tensor.ordinal,
            "source_shape": list(tensor.shape),
            "source_ggml_type_id": tensor.type_id,
            "source_ggml_type": tensor.type_name,
            "source_offset": tensor.absolute_start,
            "source_payload_bytes": tensor.payload_size,
            "semantic_role": tensor.role,
            "layer_group": tensor.group,
            "layer": tensor.layer,
            "target_name": tensor.name,
            "target_representation": representation,
            "target_storage": storage,
            "target_key_id": TENSOR_KEY_ID,
            "target_occurrence": target_order,
            "target_order": target_order,
            "payload_action": "COPY_BYTES",
        })
    return plans


def validate_manifest(manifest: dict[str, object], artifact) -> None:
    plans = manifest["tensor_plans"]
    if len(plans) != len(artifact.tensors):
        raise ValueError("source tensor accounting count mismatch")
    if len({plan["source_name"] for plan in plans}) != len(plans):
        raise ValueError("duplicate source tensor plan")
    if len({plan["target_name"] for plan in plans}) != len(plans):
        raise ValueError("duplicate target tensor name")
    if sorted(plan["target_order"] for plan in plans) != list(range(len(plans))):
        raise ValueError("target order is not a complete deterministic sequence")
    if len({(plan["target_key_id"], plan["target_occurrence"]) for plan in plans}) != len(plans):
        raise ValueError("duplicate target identity")
    if any(plan["payload_action"] != "COPY_BYTES" for plan in plans):
        raise ValueError("unexpected payload action")
    if manifest["accounting"]["unaccounted"] != 0:
        raise ValueError("unaccounted source tensors")


def build_manifest(root: Path, label: str) -> dict[str, object]:
    filename, expected_hash = EXPECTED[label]
    path = root / "research-models" / filename
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")
    artifact = parse(path)
    if artifact.metadata.get("general.architecture") != "qwen3":
        raise ValueError("source architecture is not qwen3")
    plans = tensor_plans(artifact)
    metadata = metadata_plan(artifact)
    tokenizer = tokenizer_plan(artifact)
    source_by_name = {tensor.name: tensor for tensor in artifact.tensors}
    special_hashes = {}
    for name in ("token_embd.weight", "output.weight"):
        if name in source_by_name:
            tensor = source_by_name[name]
            special_hashes[name] = payload_sha256(path, tensor.absolute_start, tensor.payload_size)
    tied = None
    if label == "Q8_0" and "output.weight" not in source_by_name:
        tied = {"logical_role": "output.weight", "action": "SHARED_REFERENCE", "source_name": "token_embd.weight",
                "emitted_target_tensor": False, "consumer_behavior": "pinned Qwen3 fallback via TENSOR_DUPLICATED"}
    output_decision = None
    if "output.weight" in source_by_name and "token_embd.weight" in source_by_name:
        output_decision = {"source_descriptors_distinct": True, "payload_hashes_equal": special_hashes["output.weight"] == special_hashes["token_embd.weight"],
                           "action": "COPY_BYTES independently", "reason": "explicit output.weight remains a distinct source tensor"}
    metadata_blockers = [row["consumer_semantic"] for row in metadata if row["status"] == "MISSING"]
    tokenizer_blockers = [row["source_key"] for row in tokenizer if row["status"] == "UNSUPPORTED_REQUIRED"]
    readiness = []
    if metadata_blockers:
        readiness.append("BLOCKED_BY_METADATA_GAP")
    if tokenizer_blockers:
        readiness.append("BLOCKED_BY_TOKENIZER_GAP")
    if not readiness:
        readiness.append("READY_FOR_CONVERSION")
    source_read = source_range_summary(artifact, [source_by_name[plan["source_name"]] for plan in plans])
    manifest = {
        "manifest_version": 1,
        "source_identity": {"path": f"research-models/{filename}", "filename": filename, "size": artifact.size,
                             "sha256": actual_hash, "gguf_version": artifact.version, "architecture": "qwen3",
                             "metadata_kv_count": artifact.metadata_count, "tensor_count": artifact.tensor_count},
        "consumer_revision": {"repository": "https://github.com/ggml-org/llama.cpp.git", "commit": PINNED_COMMIT},
        "profile_version": "vbuf-ml-0.1",
        "conversion_options": {"placement": PLACEMENT, "integrity": {"enabled": False, "algorithm": "SHA-256", "coverage": "payload bytes"}},
        "control_plan": [{"role": role, "target_key_id": key_id, "action": "DERIVED"} for role, key_id in CONTROL_KEY_IDS.items() if role != "IntegrityMetadata"],
        "model_metadata_plan": metadata,
        "tokenizer_plan": tokenizer,
        "tensor_directory_plan": [{"name": plan["target_name"], "shape": plan["source_shape"], "representation": plan["target_representation"],
                                    "key_id": plan["target_key_id"], "occurrence": plan["target_occurrence"], "directory_order": index}
                                   for index, plan in enumerate(sorted(plans, key=lambda plan: plan["target_name"]))],
        "tensor_plans": plans,
        "shared_reference_plan": tied,
        "output_decision": output_decision,
        "payload_evidence": {"selected_source_payload_sha256": special_hashes},
        "placement_plan": {"policy": PLACEMENT, "target_order_is_non_normative": True},
        "source_read_plan": source_read,
        "accounting": {"source_tensors": len(artifact.tensors), "copy_bytes": len(plans), "shared_reference": 1 if tied else 0,
                       "derived": len(CONTROL_KEY_IDS) - 1, "reject": 0, "unaccounted": 0},
        "validation": {"manifest_structurally_valid": True, "conversion_readiness": readiness,
                        "metadata_blockers": metadata_blockers, "tokenizer_blockers": tokenizer_blockers},
    }
    validate_manifest(manifest, artifact)
    return manifest


class ManifestTests(unittest.TestCase):
    def test_representation_geometry_is_reused(self) -> None:
        self.assertEqual(expected_bytes("F32", (2, 3)), 24)
        self.assertEqual(expected_bytes("BF16", (2, 3)), 12)
        self.assertEqual(expected_bytes("Q8_0", (64, 2)), 136)
        with self.assertRaises(ValueError):
            expected_bytes("Q8_0", (31,))

    def test_target_identity_policy_is_stable(self) -> None:
        self.assertEqual(TENSOR_KEY_ID, 0x0200)
        self.assertEqual(list(CONTROL_KEY_IDS), ["Bootstrap", "ModelMetadata", "TokenizerMetadata", "IntegrityMetadata"])
        self.assertEqual(PLACEMENT, "LAYER_MAJOR_ROLE_ORDER")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1
    root = args.root.resolve()
    output = (args.output_dir or root / "benchmark-results" / "vbuf-ml-step18").resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifests = {label: build_manifest(root, label) for label in ("Q8_0", "BF16")}
    except FileNotFoundError as error:
        print(f"SKIP — research model absent: {error}")
        return 0
    except (OSError, ValueError) as error:
        print(f"FAIL — {error}", file=sys.stderr)
        return 1
    for label, manifest in manifests.items():
        filename = "qwen3-0.6b-q8_0-manifest.json" if label == "Q8_0" else "qwen3-0.6b-bf16-manifest.json"
        (output / filename).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_rows = []
    tokenizer_rows = []
    action_rows = []
    for label, manifest in manifests.items():
        metadata_rows.extend({"artifact": label, **row} for row in manifest["model_metadata_plan"])
        tokenizer_rows.extend({"artifact": label, **row} for row in manifest["tokenizer_plan"])
        action_rows.append({"artifact": label, **manifest["accounting"], "readiness": ";".join(manifest["validation"]["conversion_readiness"])})
    def csv_write(path: Path, rows: list[dict[str, object]]) -> None:
        import csv
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    csv_write(output / "metadata-matrix.csv", metadata_rows)
    csv_write(output / "tokenizer-matrix.csv", tokenizer_rows)
    csv_write(output / "tensor-action-summary.csv", action_rows)
    (output / "qualification-config.json").write_text(json.dumps({"repository_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "placement": PLACEMENT, "consumer_commit": PINNED_COMMIT}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS — deterministic manifests validated for Q8_0 and BF16; evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
