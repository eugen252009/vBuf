#!/usr/bin/env python3
"""Build a metadata-only Phi-4-mini portability prototype.

This intentionally consumes pinned Hugging Face configuration and
safetensors-header evidence represented below; it never downloads or executes
model weights. The descriptor reuses the P0 JSON vocabulary and adds only
generic storage views, aliases, position attributes, and KV state attributes.
"""
from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from typing import Any


PHI_REPOSITORY = "microsoft/Phi-4-mini-instruct"
PHI_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
PHI_CONFIG_REVISION = PHI_REVISION
PHI_TOKENIZER_REVISION = PHI_REVISION

# Exact values from config.json and the layer-0 safetensors headers at the
# pinned revision. No payload bytes are needed for this boundary experiment.
CONFIG: dict[str, Any] = {
    "architecture": "Phi3ForCausalLM", "model_type": "phi3", "parameter_count": 3_836_021_760,
    "vocab_size": 200064, "layer_count": 32, "hidden_size": 3072,
    "attention_heads": 24, "kv_heads": 8, "head_dimension": 128,
    "intermediate_size": 8192, "max_context": 131072, "original_context": 4096,
    "rope_theta": 10000.0, "partial_rotary_factor": 0.75, "rotary_dimensions": 96,
    "sliding_window": 262144, "weight_tying": True, "hidden_act": "silu",
    "dtype": "BF16",
    "long_factor": [1, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897,
                    1.956131817, 2.187582649, 2.446418898, 2.735880826, 3.059592084,
                    3.421605075, 3.826451687, 4.279200023, 4.785517845, 5.351743533,
                    5.984965424, 6.693110555, 7.485043894, 8.370679318, 9.36110372,
                    10.4687158, 11.70738129, 13.09260651, 14.64173252, 16.37415215,
                    18.31155283, 20.47818807, 22.90118105, 25.61086418, 28.64115884,
                    32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7,
                    32.71, 32.93, 32.97, 33.28, 33.49, 33.5, 44.16, 47.77],
    "short_factor": [1.0] * 48,
}

TENSORS = {
    "token_embedding": ("model.embed_tokens.weight", [200064, 3072], "model-00001-of-00002.safetensors"),
    "input_norm": ("model.layers.0.input_layernorm.weight", [3072], "model-00001-of-00002.safetensors"),
    "qkv": ("model.layers.0.self_attn.qkv_proj.weight", [5120, 3072], "model-00001-of-00002.safetensors"),
    "attention_output": ("model.layers.0.self_attn.o_proj.weight", [3072, 3072], "model-00001-of-00002.safetensors"),
    "post_attention_norm": ("model.layers.0.post_attention_layernorm.weight", [3072], "model-00001-of-00002.safetensors"),
    "gate_up": ("model.layers.0.mlp.gate_up_proj.weight", [16384, 3072], "model-00001-of-00002.safetensors"),
    "down": ("model.layers.0.mlp.down_proj.weight", [3072, 8192], "model-00001-of-00002.safetensors"),
}


def binding(semantic: str, tensor_key: str, view: dict[str, Any] | None = None) -> dict[str, Any]:
    name, shape, shard = TENSORS[tensor_key]
    result: dict[str, Any] = {
        "semantic": semantic,
        "tensor_ref": {"directory": "TensorDirectory", "semantic_key": tensor_key},
        "shape": shape,
        "representation": {"kind": "bf16", "element_type": "bf16"},
        "source_provenance": {"name": name, "shard": shard, "repository": PHI_REPOSITORY, "revision": PHI_REVISION},
    }
    if view is not None:
        result["view"] = view
    return result


def build_sidecar() -> dict[str, Any]:
    bindings = [
        binding("embedding.token", "token_embedding"),
        binding("layer.0.attention.input_norm", "input_norm"),
        binding("layer.0.attention.query_weight", "qkv", {"axis": 0, "start": 0, "length": 3072}),
        binding("layer.0.attention.key_weight", "qkv", {"axis": 0, "start": 3072, "length": 1024}),
        binding("layer.0.attention.value_weight", "qkv", {"axis": 0, "start": 4096, "length": 1024}),
        binding("layer.0.attention.output_weight", "attention_output"),
        binding("layer.0.mlp.input_norm", "post_attention_norm"),
        binding("layer.0.mlp.gate_weight", "gate_up", {"axis": 0, "start": 0, "length": 8192}),
        binding("layer.0.mlp.up_weight", "gate_up", {"axis": 0, "start": 8192, "length": 8192}),
        binding("layer.0.mlp.down_weight", "down"),
    ]
    return {
        "sidecar": "vbuf-ml.phi3.dense-typed-program",
        "version": 1,
        "source": {"repository": PHI_REPOSITORY, "revision": PHI_REVISION,
                    "config_revision": PHI_CONFIG_REVISION, "tokenizer_revision": PHI_TOKENIZER_REVISION},
        "portable_boundary": {"excludes": ["safetensors_offsets_as_identity", "backend_buffers", "device_placement",
                                             "source_read_plan", "runtime_schedule", "cache_contents"],
                              "identity": "semantic TensorRef plus generic storage view/alias"},
        "model_parameters": CONFIG,
        "tensor_bindings": bindings,
        "aliases": [{"semantic": "lm_head.weight", "target": "embedding.token",
                      "relation": "same_storage", "reason": "config.tie_word_embeddings and absent lm_head tensor"}],
        "program": [
            {"id": "embedding", "kind": "embedding", "inputs": ["token_ids"], "outputs": ["hidden"],
             "weights": ["embedding.token"]},
            {"id": "layer.0.attention", "kind": "attention", "inputs": ["hidden", "position", "kv_read"],
             "outputs": ["attention_output", "kv_write"], "weights": ["layer.0.attention.input_norm",
             "layer.0.attention.query_weight", "layer.0.attention.key_weight", "layer.0.attention.value_weight",
             "layer.0.attention.output_weight"], "attributes": {"heads": 24, "kv_heads": 8,
             "head_dimension": 128, "causal": True, "scale": "derived:head_dimension^-0.5",
             "mask_window": 262144, "position_transform": "position.longrope.partial_rotary"}},
            {"id": "layer.0.mlp", "kind": "composed_dense_mlp", "inputs": ["attention_residual"],
             "outputs": ["mlp_output"], "weights": ["layer.0.mlp.input_norm", "layer.0.mlp.gate_weight",
             "layer.0.mlp.up_weight", "layer.0.mlp.down_weight"], "composition": ["rms_norm", "matmul_gate_up",
             "split_gate_up", "silu", "mul", "matmul_down", "residual_add"]},
        ],
        "position_schema": {"ref": "position.longrope.partial_rotary", "kind": "rotary_position_transform",
            "rotary_dimensions": 96, "head_dimension": 128, "theta": 10000.0, "mode": "longrope",
            "original_context": 4096, "max_context": 131072, "short_factor": CONFIG["short_factor"],
            "long_factor": CONFIG["long_factor"], "switch": "sequence_length > original_context"},
        "state_schema": [
            {"ref": "layer.0.kv.key", "kind": "kv_key", "shape": [8, "sequence_length", 128],
             "scope": "layer", "lifetime": "sequence", "access": "read-write",
             "transition": "append(position, key); read(position_range)"},
            {"ref": "layer.0.kv.value", "kind": "kv_value", "shape": [8, "sequence_length", 128],
             "scope": "layer", "lifetime": "sequence", "access": "read-write",
             "transition": "append(position, value); read(position_range)"},
            {"ref": "sequence.position", "kind": "position", "shape": [1], "element_type": "u64",
             "scope": "sequence", "lifetime": "sequence", "access": "read-write",
             "transition": "position_next = position + token_count", "reset": "sequence_start"},
        ],
        "tokenizer": {"kind": "gpt2_bpe", "vocab_size": 200064, "tokenizer_class": "GPT2Tokenizer",
                      "vocabulary": "vocab.json", "merges": "merges.txt", "special_tokens": [199999, 200018,
                      200019, 200020, 200021, 200022, 200028], "bos": False, "eos": False,
                      "chat_template": "application_profile_only"},
    }


def validate(sidecar: dict[str, Any]) -> None:
    bindings = sidecar["tensor_bindings"]
    if len({item["semantic"] for item in bindings}) != len(bindings):
        raise ValueError("duplicate semantic binding")
    storage = {item["tensor_ref"]["semantic_key"] for item in bindings}
    for item in bindings:
        storage_key = item["tensor_ref"]["semantic_key"]
        if storage_key not in TENSORS:
            raise ValueError("unknown storage binding")
        if "view" in item:
            storage_shape = TENSORS[storage_key][1]
            if item["view"]["start"] < 0 or item["view"]["length"] <= 0 or item["view"]["start"] + item["view"]["length"] > storage_shape[0]:
                raise ValueError("invalid view")
    alias = sidecar["aliases"][0]
    if alias["target"] not in {item["semantic"] for item in bindings} or alias["target"] not in {"embedding.token"}:
        raise ValueError("invalid tied alias")
    if len(storage) != 7:
        raise ValueError("unexpected representative storage count")
    if sidecar["position_schema"]["rotary_dimensions"] != 96:
        raise ValueError("partial rotary semantics missing")


class PhiSidecarTests(unittest.TestCase):
    def test_dense_phi_layer_uses_optional_features_without_placeholders(self) -> None:
        sidecar = build_sidecar()
        validate(sidecar)
        kinds = {op["kind"] for op in sidecar["program"]}
        self.assertEqual(kinds, {"embedding", "attention", "composed_dense_mlp"})
        self.assertNotIn("ssm", json.dumps(sidecar["program"]))
        self.assertNotIn("moe", json.dumps(sidecar["program"]))

    def test_fused_storage_and_tied_embedding_are_explicit(self) -> None:
        sidecar = build_sidecar()
        qkv = [item for item in sidecar["tensor_bindings"] if item["tensor_ref"]["semantic_key"] == "qkv"]
        self.assertEqual(len(qkv), 3)
        self.assertEqual(sidecar["aliases"][0]["relation"], "same_storage")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "research/results/vbuf-import-boundary-audit/phi4-mini-typed-program.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[__file__], exit=False).result.wasSuccessful() else 1
    sidecar = build_sidecar()
    validate(sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS — Phi typed sidecar written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
