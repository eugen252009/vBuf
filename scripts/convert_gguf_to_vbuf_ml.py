#!/usr/bin/env python3
"""Manifest-driven GGUF -> vBuf-ML conversion orchestrator.

Policy and tokenizer resolution come from the qualified JSON manifest. The
Rust helper is the sole production canonical-v0.6 writer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_step18_manifest import CONTROL_KEY_IDS, TENSOR_KEY_ID  # noqa: E402
from qualify_step16 import EXPECTED, parse  # noqa: E402
from qualify_step17 import PINNED_COMMIT  # noqa: E402

PLAN_MAGIC = b"VBUF20PL"
PLAN_VERSION = 2
REPRESENTATIONS = {"CanonicalPrimitive": 0, "BF16": 1, "GGML_Q8_0": 2, "GGML_Q4_0": 3, "GGML_Q2_K": 4, "GGML_IQ1_S": 5, "GGML_Q4_K": 6, "GGML_IQ4_NL": 7, "GGML_IQ4_XS": 8, "GGML_Q3_K": 9, "GGML_IQ2_XXS": 10, "GGML_IQ2_XS": 11, "GGML_IQ2_S": 12, "GGML_Q5_K": 13, "GGML_Q6_K": 14}
METADATA_IDS = {"Architecture": 1, "ContextLength": 2, "EmbeddingLength": 3, "LayerCount": 4,
                "HeadCount": 5, "FeedForwardLength": 6, "NormalizationEpsilon": 7, "RopeTheta": 8,
                "KVHeadCount": 9, "KeyHeadDimension": 10, "ValueHeadDimension": 11}
ROLE_IDS = {"TokenTextBytes": 1, "TokenOffsets": 2, "TokenScores": 3, "TokenTypes": 4,
            "BosId": 5, "EosId": 6, "UnkId": 7, "PadId": 8, "MergeLeftIds": 9,
            "MergeRightIds": 10, "TokenizerModelIdentity": 11, "PreTokenizerIdentity": 12,
            "AddBos": 13, "ChatTemplate": 14}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def u8(out: bytearray, value: int) -> None: out.extend(struct.pack("<B", value))
def u16(out: bytearray, value: int) -> None: out.extend(struct.pack("<H", value))
def u32(out: bytearray, value: int) -> None: out.extend(struct.pack("<I", value))
def u64(out: bytearray, value: int) -> None: out.extend(struct.pack("<Q", value))
def blob(out: bytearray, value: bytes) -> None: u64(out, len(value)); out.extend(value)
def string(out: bytearray, value: str) -> None: blob(out, value.encode("utf-8"))


def build_plan(source: Path, manifest: dict, artifact, integrity: str) -> bytes:
    identity = manifest["source_identity"]
    if identity["size"] != source.stat().st_size or identity["sha256"] != digest(source):
        raise ValueError("source identity does not match manifest")
    if manifest["profile_version"] != "vbuf-ml-0.1" or manifest["consumer_revision"]["commit"] != PINNED_COMMIT:
        raise ValueError("manifest profile or pinned consumer revision is stale")
    if manifest["validation"]["conversion_readiness"] != ["READY_FOR_CONVERSION"]:
        raise ValueError("manifest is not conversion-ready")
    if manifest["conversion_options"]["placement"] not in {"LAYER_MAJOR_ROLE_ORDER", "DEEPSEEK_HOT_ROLE_ORDER"} or not 3 <= int(manifest["conversion_options"].get("base_shift", -1)) <= 8:
        raise ValueError("manifest placement or BaseShift policy is not qualified")
    if integrity != "none":
        raise ValueError("Step 20 currently supports only --integrity none")

    by_name = {tensor.name: tensor for tensor in artifact.tensors}
    plans = sorted(manifest["tensor_plans"], key=lambda row: row["target_order"])
    if len(plans) != len(by_name): raise ValueError("manifest tensor count mismatch")
    for row in plans:
        source_tensor = by_name.get(row["source_name"])
        if source_tensor is None or row["target_name"] != row["source_name"]:
            raise ValueError(f"manifest tensor name mismatch: {row['source_name']}")
        if row["source_offset"] != source_tensor.absolute_start or row["source_payload_bytes"] != source_tensor.payload_size:
            raise ValueError(f"manifest source range mismatch: {row['source_name']}")
        if row["target_key_id"] != TENSOR_KEY_ID or row["target_occurrence"] != row["target_order"]:
            raise ValueError("manifest tensor identity mismatch")
        if row["payload_action"] != "COPY_BYTES" or row["target_representation"] not in REPRESENTATIONS:
            raise ValueError("manifest tensor action or representation is unsupported")

    tokenizer = manifest["tokenizer_conversion_plan"]
    if tokenizer["kind"] != "Gpt2BpeQwen2" or tokenizer["model_id"] != 1 or tokenizer["pre_tokenizer_id"] not in {1, 2}:
        raise ValueError("unsupported tokenizer conversion plan")
    tokens = artifact.metadata["tokenizer.ggml.tokens"]
    token_types = artifact.metadata["tokenizer.ggml.token_type"]
    merges = artifact.metadata["tokenizer.ggml.merges"]
    left = tokenizer["merge_left_ids"]; right = tokenizer["merge_right_ids"]
    if len(tokens) != len(token_types) or len(merges) != len(left) or len(left) != len(right):
        raise ValueError("manifest tokenizer counts do not match source")

    out = bytearray(PLAN_MAGIC)
    u32(out, PLAN_VERSION); u8(out, manifest["conversion_options"]["base_shift"]); u8(out, 0); u16(out, 0)
    u64(out, source.stat().st_size); out.extend(bytes.fromhex(identity["sha256"]))
    # ModelMetadata values are copied from the manifest's already-qualified plan.
    metadata_rows = [row for row in manifest["model_metadata_plan"] if row["target"] in METADATA_IDS]
    u16(out, len(metadata_rows))
    for row in sorted(metadata_rows, key=lambda item: METADATA_IDS[item["target"]]):
        key_id = METADATA_IDS[row["target"]]; value = row["source_value"]
        u16(out, key_id); u8(out, int(key_id in {1, 2, 3, 4, 5}));
        if key_id == 1: u8(out, 1); blob(out, str(value).encode("utf-8"))
        elif key_id in {7, 8}: u8(out, 3); blob(out, struct.pack("<d", float(value)))
        else: u8(out, 2); blob(out, struct.pack("<Q", int(value)))

    text = b"".join(token.encode("utf-8") for token in tokens)
    offsets = [0];
    for token in tokens: offsets.append(offsets[-1] + len(token.encode("utf-8")))
    blob(out, text); u64(out, len(offsets)); [u64(out, value) for value in offsets]
    u64(out, len(token_types)); [u32(out, int(value)) for value in token_types]
    u64(out, len(left)); [u32(out, int(value)) for value in left]; [u32(out, int(value)) for value in right]
    u8(out, int(bool(tokenizer["add_bos"])))
    u8(out, int(tokenizer["pre_tokenizer_id"]))
    specials = [(ROLE_IDS["BosId"], artifact.metadata["tokenizer.ggml.bos_token_id"]),
                (ROLE_IDS["EosId"], artifact.metadata["tokenizer.ggml.eos_token_id"]),
                (ROLE_IDS["PadId"], artifact.metadata["tokenizer.ggml.padding_token_id"])]
    u8(out, len(specials));
    for role, value in specials: u8(out, role); u64(out, int(value))
    blob(out, artifact.metadata["tokenizer.chat_template"].encode("utf-8"))

    u32(out, len(plans))
    for row in plans:
        string(out, row["target_name"]); u8(out, len(row["source_shape"]));
        for dimension in row["source_shape"]: u64(out, dimension)
        u8(out, REPRESENTATIONS[row["target_representation"]]); u64(out, row["source_offset"]); u64(out, row["source_payload_bytes"]); u32(out, row["target_order"])
    moe = manifest.get("moe_plan")
    if moe is None:
        u8(out, 0)
    else:
        u8(out, 1)
        for key in ("expert_count", "active_expert_count", "layer_count", "shared_expert_count"):
            u32(out, int(moe[key]))
        u8(out, int(bool(moe["shared_experts"])))
    return bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--integrity", choices=("none", "sha256"), default="none")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    args = parser.parse_args()
    source, manifest_path, target = args.source.resolve(), args.manifest.resolve(), args.target.resolve()
    if not source.exists() or not manifest_path.exists():
        print("SKIP — source or manifest absent")
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = parse(source)
        plan = build_plan(source, manifest, artifact, args.integrity)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FAIL — manifest/source validation: {error}", file=sys.stderr); return 1
    evidence = (args.evidence_dir or source.parent.parent / "benchmark-results" / "vbuf-ml-step20").resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".plan", dir=target.parent, delete=False) as stream:
        plan_path = Path(stream.name); stream.write(plan)
    temporary_target = target.with_name(target.name + ".tmp")
    try:
        command = ["cargo", "run", "--quiet", "--manifest-path", str(source.parent.parent / "rust" / "Cargo.toml"), "-p", "vbuf-ml", "--bin", "vbuf-ml-convert", "--", "--source", str(source), "--plan", str(plan_path), "--target", str(temporary_target), "--evidence", str(evidence)]
        result = subprocess.run(command, cwd=source.parent.parent, text=True)
        if result.returncode != 0: return result.returncode
        os.replace(temporary_target, target)
        target_hash = digest(target)
        summary = {"artifact": target.name, "source": source.name, "source_sha256": digest(source), "target_sha256": target_hash, "target_size": target.stat().st_size, "integrity": args.integrity, "tool": "vbuf-ml-convert", "manifest": manifest_path.name}
        (evidence / (target.stem + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"PASS — converted and independently validated: {target}")
        return 0
    finally:
        plan_path.unlink(missing_ok=True); temporary_target.unlink(missing_ok=True)


if __name__ == "__main__": raise SystemExit(main())
