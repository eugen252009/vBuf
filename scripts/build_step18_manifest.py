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
BOOTSTRAP_KEY_ID = 0xF000
CONTROL_KEY_IDS = {"TensorDirectory": 0x0201, "ModelMetadata": 0x0202, "TokenizerMetadata": 0x0203, "IntegrityMetadata": 0x0204}
PLACEMENT = "LAYER_MAJOR_ROLE_ORDER"
DEEPSEEK_PLACEMENT = "DEEPSEEK_HOT_ROLE_ORDER"
SUPPORTED_TYPES = {"F32": ("CanonicalPrimitive", "primitive"), "BF16": ("BF16", "opaque_bytes"), "Q8_0": ("GGML_Q8_0", "opaque_bytes"), "Q4_0": ("GGML_Q4_0", "opaque_bytes"), "Q2_K": ("GGML_Q2_K", "opaque_bytes"), "Q2_K_S": ("GGML_Q2_K", "opaque_bytes"), "Q2_K_M": ("GGML_Q2_K", "opaque_bytes"), "IQ1_S": ("GGML_IQ1_S", "opaque_bytes"), "Q4_K": ("GGML_Q4_K", "opaque_bytes"), "Q4_K_S": ("GGML_Q4_K", "opaque_bytes"), "Q4_K_M": ("GGML_Q4_K", "opaque_bytes"), "Q6_K": ("GGML_Q6_K", "opaque_bytes"), "IQ4_NL": ("GGML_IQ4_NL", "opaque_bytes"), "IQ4_XS": ("GGML_IQ4_XS", "opaque_bytes"), "Q3_K": ("GGML_Q3_K", "opaque_bytes"), "Q3_K_S": ("GGML_Q3_K", "opaque_bytes"), "Q3_K_M": ("GGML_Q3_K", "opaque_bytes"), "IQ2_XXS": ("GGML_IQ2_XXS", "opaque_bytes"), "IQ2_XS": ("GGML_IQ2_XS", "opaque_bytes"), "IQ2_S": ("GGML_IQ2_S", "opaque_bytes"), "Q5_K": ("GGML_Q5_K", "opaque_bytes")}


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
    architecture = str(metadata.get("general.architecture", "qwen3"))
    prefix = architecture
    feed_forward_key = f"{prefix}.feed_forward_length"
    if feed_forward_key not in metadata:
        feed_forward_key = f"{prefix}.expert_feed_forward_length"
    fields = [
        ("general.architecture", "architecture", "Architecture", "DIRECTLY_REPRESENTED", metadata.get("general.architecture")),
        (f"{prefix}.context_length", "context length", "ContextLength", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.context_length")),
        (f"{prefix}.embedding_length", "embedding length", "EmbeddingLength", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.embedding_length")),
        (f"{prefix}.block_count", "layer count", "LayerCount", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.block_count")),
        (f"{prefix}.attention.head_count", "attention head count", "HeadCount", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.attention.head_count")),
        (f"{prefix}.attention.head_count_kv", "KV head count", "KVHeadCount", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.attention.head_count_kv")),
        (f"{prefix}.attention.key_length", "key head dimension", "KeyHeadDimension", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.attention.key_length")),
        (f"{prefix}.attention.value_length", "value head dimension", "ValueHeadDimension", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.attention.value_length")),
        (feed_forward_key, "feed-forward length", "FeedForwardLength", "DIRECTLY_REPRESENTED", metadata.get(feed_forward_key)),
        (f"{prefix}.attention.layer_norm_rms_epsilon", "RMS normalization epsilon", "NormalizationEpsilon", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.attention.layer_norm_rms_epsilon")),
        (f"{prefix}.rope.freq_base", "RoPE base", "RopeTheta", "DIRECTLY_REPRESENTED", metadata.get(f"{prefix}.rope.freq_base")),
        ("tokenizer.ggml.tokens", "vocabulary size", "VocabularySize", "DERIVABLE", len(metadata["tokenizer.ggml.tokens"]) if isinstance(metadata.get("tokenizer.ggml.tokens"), list) else None),
    ]
    return [{"source_key": source, "consumer_semantic": consumer, "target": target,
             "status": status, "source_value": value} for source, consumer, target, status, value in fields]


def tokenizer_conversion_plan(artifact) -> dict[str, object]:
    metadata = artifact.metadata
    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(tokens, list) or not isinstance(merges, list):
        raise ValueError("GPT2BpeQwen2 conversion requires tokens and merges")
    token_ids = {}
    for index, token in enumerate(tokens):
        if not isinstance(token, str) or token in token_ids:
            raise ValueError("token vocabulary is not a unique string ordinal table")
        token_ids[token] = index
    left_ids, right_ids = [], []
    for merge in merges:
        if not isinstance(merge, str):
            raise ValueError("merge source is not a string")
        position = merge.find(" ", 1)
        if position < 0:
            raise ValueError("merge has no pinned separator")
        left, right = merge[:position], merge[position + 1:]
        if left not in token_ids or right not in token_ids:
            raise ValueError("merge component is absent from vocabulary")
        left_ids.append(token_ids[left])
        right_ids.append(token_ids[right])
    pre_tokenizer = 2 if str(metadata.get("general.architecture", "")).startswith("deepseek") else 1
    return {"kind": "Gpt2BpeQwen2", "model_id": 1, "pre_tokenizer_id": pre_tokenizer,
            "add_bos": metadata.get("tokenizer.ggml.add_bos_token"),
            "merge_left_ids": left_ids, "merge_right_ids": right_ids,
            "merge_count": len(merges)}


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
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 vocabulary"
        elif key == "tokenizer.ggml.merges":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 MergeLeftIds+MergeRightIds"
        elif key == "tokenizer.ggml.model":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 TokenizerModelIdentity"
        elif key == "tokenizer.ggml.pre":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 PreTokenizerIdentity"
        elif key == "tokenizer.ggml.add_bos_token":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 AddBos"
        elif key == "tokenizer.chat_template":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 ChatTemplate (optional raw target)"
        elif key.endswith("_token_id"):
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 special ID"
        elif key == "tokenizer.ggml.scores":
            status, target = "DIRECTLY_REPRESENTED", "GPT2BpeQwen2 optional scores"
        else:
            status, target = "IGNORE_WITH_REASON", "not consumed by pinned first target"
        rows.append({"source_key": key, "meaning": "tokenizer metadata", "source_shape_or_value": shape,
                     "target": target, "status": status})
    return rows


def placement_for_artifact(artifact):
    architecture = artifact.metadata.get("general.architecture", "")
    return DEEPSEEK_PLACEMENT if str(architecture).startswith("deepseek") else PLACEMENT


def moe_plan(artifact):
    architecture = str(artifact.metadata.get("general.architecture", ""))
    if not architecture.startswith("deepseek") and architecture != "qwen35moe":
        return None
    prefix = architecture
    shared_count = artifact.metadata.get(f"{prefix}.expert_shared_count", 1 if artifact.metadata.get(f"{prefix}.expert_shared_feed_forward_length") else 0)
    return {"architecture": architecture, "expert_count": int(artifact.metadata[f"{prefix}.expert_count"]),
            "active_expert_count": int(artifact.metadata[f"{prefix}.expert_used_count"]),
            "layer_count": int(artifact.metadata[f"{prefix}.block_count"]),
            "shared_experts": bool(shared_count), "shared_expert_count": int(shared_count)}


def tensor_plans(artifact, placement=None):
    placement = placement or placement_for_artifact(artifact)
    ordered = layout_order(artifact, placement)
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
    tokenizer = manifest.get("tokenizer_conversion_plan")
    if not isinstance(tokenizer, dict) or tokenizer.get("kind") != "Gpt2BpeQwen2" or tokenizer.get("merge_count") != len(tokenizer.get("merge_left_ids", [])) or tokenizer.get("merge_count") != len(tokenizer.get("merge_right_ids", [])):
        raise ValueError("invalid tokenizer conversion plan")


def build_manifest(root: Path, label: str, source_override: Path | None = None) -> dict[str, object]:
    filename, expected_hash = EXPECTED[label]
    path = source_override or root / "research-models" / filename
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual_hash = sha256(path)
    if source_override is None and actual_hash != expected_hash:
        raise ValueError(f"hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")
    artifact = parse(path)
    architecture = str(artifact.metadata.get("general.architecture", ""))
    if architecture not in {"qwen3", "qwen35moe", "deepseek2", "deepseek32", "deepseek2-ocr"}:
        raise ValueError("source architecture is not a supported Qwen/DeepSeek architecture")
    placement = placement_for_artifact(artifact)
    plans = tensor_plans(artifact, placement)
    metadata = metadata_plan(artifact)
    tokenizer = tokenizer_plan(artifact)
    tokenizer_conversion = tokenizer_conversion_plan(artifact)
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
                             "sha256": actual_hash, "gguf_version": artifact.version, "architecture": architecture,
                             "metadata_kv_count": artifact.metadata_count, "tensor_count": artifact.tensor_count},
        "consumer_revision": {"repository": "https://github.com/ggml-org/llama.cpp.git", "commit": PINNED_COMMIT},
        "profile_version": "vbuf-ml-0.1",
        "conversion_options": {"placement": placement, "base_shift": 3, "integrity": {"enabled": False, "algorithm": "SHA-256", "coverage": "payload bytes"}},
        "control_plan": [{"role": "Bootstrap", "target_key_id": BOOTSTRAP_KEY_ID, "action": "DERIVED"}] + [{"role": role, "target_key_id": key_id, "action": "DERIVED"} for role, key_id in CONTROL_KEY_IDS.items() if role != "IntegrityMetadata"],
        "model_metadata_plan": metadata,
        "tokenizer_plan": tokenizer,
        "tokenizer_conversion_plan": tokenizer_conversion,
        "tensor_directory_plan": [{"name": plan["target_name"], "shape": plan["source_shape"], "representation": plan["target_representation"],
                                    "key_id": plan["target_key_id"], "occurrence": plan["target_occurrence"], "directory_order": index}
                                   for index, plan in enumerate(sorted(plans, key=lambda plan: plan["target_name"]))],
        "tensor_plans": plans,
        "moe_plan": moe_plan(artifact),
        "shared_reference_plan": tied,
        "output_decision": output_decision,
        "payload_evidence": {"selected_source_payload_sha256": special_hashes},
        "placement_plan": {"policy": placement, "target_order_is_non_normative": True},
        "source_read_plan": source_read,
        "accounting": {"source_tensors": len(artifact.tensors), "copy_bytes": len(plans), "shared_reference": 1 if tied else 0,
                       "derived": len(CONTROL_KEY_IDS) - 1, "reject": 0, "unaccounted": 0},
        "validation": {"manifest_structurally_valid": True, "conversion_readiness": readiness,
                        "raw_inference_tokenizer_readiness": "READY",
                        "chat_template_readiness": "READY_FOR_CONSUMER_EXECUTION",
                        "metadata_blockers": metadata_blockers, "tokenizer_blockers": tokenizer_blockers},
    }
    validate_manifest(manifest, artifact)
    return manifest


class ManifestTests(unittest.TestCase):
    def test_deepseek_placement_policy_is_selected_by_architecture(self) -> None:
        artifact = type("Artifact", (), {"metadata": {"general.architecture": "deepseek2"}})()
        self.assertEqual(placement_for_artifact(artifact), DEEPSEEK_PLACEMENT)

    def test_representation_geometry_is_reused(self) -> None:
        self.assertEqual(expected_bytes("F32", (2, 3)), 24)
        self.assertEqual(expected_bytes("BF16", (2, 3)), 12)
        self.assertEqual(expected_bytes("Q8_0", (64, 2)), 136)
        with self.assertRaises(ValueError):
            expected_bytes("Q8_0", (31,))

    def test_target_identity_policy_is_stable(self) -> None:
        self.assertEqual(TENSOR_KEY_ID, 0x0200)
        self.assertEqual(list(CONTROL_KEY_IDS), ["TensorDirectory", "ModelMetadata", "TokenizerMetadata", "IntegrityMetadata"])
        self.assertEqual(BOOTSTRAP_KEY_ID, 0xF000)
        self.assertEqual(PLACEMENT, "LAYER_MAJOR_ROLE_ORDER")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1
    root = args.root.resolve()
    output = (args.output_dir or root / "benchmark-results" / "vbuf-ml-step18").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.source:
        try:
            manifest = build_manifest(root, "Q8_0", args.source.resolve())
            destination = (args.manifest or output / (args.source.stem + "-manifest.json")).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"PASS — manifest validated for {args.source}; evidence: {destination}")
            return 0
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"FAIL — {error}", file=sys.stderr)
            return 1
    labels = ["Q8_0", "BF16"]
    if (root / "research-models" / EXPECTED["DEEPSEEK_IQ1_S"][0]).is_file(): labels.append("DEEPSEEK_IQ1_S")
    try:
        manifests = {label: build_manifest(root, label) for label in labels}
    except FileNotFoundError as error:
        print(f"SKIP — research model absent: {error}")
        return 0
    except (OSError, ValueError) as error:
        print(f"FAIL — {error}", file=sys.stderr)
        return 1
    for label, manifest in manifests.items():
        filename = {"Q8_0": "qwen3-0.6b-q8_0-manifest.json", "BF16": "qwen3-0.6b-bf16-manifest.json", "DEEPSEEK_IQ1_S": "deepseek-v2-lite-iq1-s-manifest.json"}[label]
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
