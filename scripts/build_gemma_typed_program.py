#!/usr/bin/env python3
"""Build a metadata-only Gemma 3 semantic compatibility prototype.

The official checkpoint is gated, so this uses the official Hub model/revision
and file manifest plus the public Gemma3 implementation. The exact text config
values are cross-checked against a public derivative without downloading model
weights. No execution is performed.
"""
from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from typing import Any


MODEL = "google/gemma-3-270m-it"
REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
CONFIG_FILE_OID = "48bf8ed00bfb682ff0d4713914e9934c85caae4f"

CONFIG: dict[str, Any] = {
    "architecture": "Gemma3ForCausalLM",
    "model_type": "gemma3_text",
    "parameter_count": 268098176,
    "vocab_size": 262144,
    "layer_count": 18,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "attention_heads": 4,
    "kv_heads": 1,
    "head_dimension": 256,
    "max_context": 32768,
    "local_window": 512,
    "sliding_window_pattern": 6,
    "rope_theta_global": 1000000.0,
    "rope_theta_local": 10000.0,
    "rms_norm_epsilon": 1e-6,
    "query_pre_attention_scalar": 256,
    "activation": "gelu_pytorch_tanh",
    "attention_softcap": None,
    "logit_softcap": None,
    "weight_tying": True,
    "dtype": "BF16",
}

LAYER_TYPES = ["sliding_attention" if (index + 1) % 6 else "full_attention" for index in range(18)]


def binding(semantic: str, source_role: str) -> dict[str, Any]:
    return {"semantic": semantic, "tensor_ref": {"directory": "TensorDirectory", "semantic_key": source_role},
            "source_provenance": {"role": source_role, "repository": MODEL, "revision": REVISION},
            "representation": {"kind": "bf16", "element_type": "bf16"}}


def build_sidecar() -> dict[str, Any]:
    return {
        "sidecar": "vbuf-ml.gemma3_text.typed-program",
        "version": 1,
        "source": {"repository": MODEL, "revision": REVISION, "config_file_oid": CONFIG_FILE_OID,
                    "config_access": "official_manifest_plus_public_implementation_cross_check",
                    "weights_downloaded": False},
        "portable_boundary": {"excludes": ["vision_backend", "image_features", "backend_buffers",
                                             "device_placement", "mask_buffers", "frequency_tables", "cache_contents"],
                              "text_path": "Gemma3ForCausalLM"},
        "model_parameters": CONFIG,
        "attention_pattern": [{"layer": index, "scope": scope, "window": 512 if scope == "sliding_attention" else None,
                                "position_transform": "position.local_rope" if scope == "sliding_attention" else "position.global_rope"}
                               for index, scope in enumerate(LAYER_TYPES)],
        "tensor_bindings": [
            binding("embedding.token", "embedding.token"),
            binding("attention.query_weight", "attention.query_weight"),
            binding("attention.key_weight", "attention.key_weight"),
            binding("attention.value_weight", "attention.value_weight"),
            binding("attention.query_norm", "attention.query_norm"),
            binding("attention.key_norm", "attention.key_norm"),
            binding("attention.output_weight", "attention.output_weight"),
            binding("mlp.gate_weight", "mlp.gate_weight"),
            binding("mlp.up_weight", "mlp.up_weight"),
            binding("mlp.down_weight", "mlp.down_weight"),
        ],
        "aliases": [{"semantic": "lm_head.weight", "target": "embedding.token", "relation": "same_storage"}],
        "position_schema": [
            {"ref": "position.global_rope", "kind": "rotary_position_transform", "theta": 1000000.0,
             "head_dimension": 256, "scope": "global_attention"},
            {"ref": "position.local_rope", "kind": "rotary_position_transform", "theta": 10000.0,
             "head_dimension": 256, "scope": "sliding_attention"},
        ],
        "program": [{"id": "layer.patterned_attention_mlp", "kind": "layer_template", "for_each": "attention_pattern",
                      "composition": ["rms_norm(parameterization=one_plus_weight)", "q_proj", "k_proj", "v_proj",
                                      "q_norm", "k_norm", "position_transform(scope)", "attention(scope, causal, window)",
                                      "o_proj", "residual_add", "rms_norm(parameterization=one_plus_weight)",
                                      "gelu(gate_proj(x))", "mul(up_proj(x))", "down_proj", "rms_norm(parameterization=one_plus_weight)",
                                      "residual_add"],
                      "attributes": {"heads": 4, "kv_heads": 1, "head_dimension": 256, "causal": True,
                                     "query_scale": "derived:query_pre_attention_scalar^-0.5",
                                     "norm_parameterization": "one_plus_weight", "attention_pattern_ref": "attention_pattern"}}],
        "state_schema": [
            {"ref": "layered.kv.key", "kind": "kv_key", "scope": "layer", "lifetime": "sequence",
             "access": "read-write", "transition": "append(position, key); read(scope.window, position_range)",
             "variants": {"sliding_attention": {"window": 512}, "full_attention": {"window": None}}},
            {"ref": "layered.kv.value", "kind": "kv_value", "scope": "layer", "lifetime": "sequence",
             "access": "read-write", "transition": "append(position, value); read(scope.window, position_range)"},
            {"ref": "sequence.position", "kind": "position", "scope": "sequence", "lifetime": "sequence",
             "access": "read-write", "transition": "position_next = position + token_count", "reset": "sequence_start"},
        ],
        "tokenizer": {"kind": "sentencepiece", "model": "tokenizer.model", "vocabulary_size": 262144,
                      "bos_id": 2, "eos_id": 106, "pad_id": 0, "added_tokens": "added_tokens.json",
                      "chat_template": "chat_template.jinja", "chat_template_application_profile": True},
        "multimodal_boundary": {"selected_path": "text_only", "vision_semantics": "outside_selected_program",
                                 "image_adapter": "not imported", "language_model_input": "token_or_embedded_sequence"},
    }


def validate(sidecar: dict[str, Any]) -> None:
    if len(sidecar["attention_pattern"]) != sidecar["model_parameters"]["layer_count"]:
        raise ValueError("attention pattern does not cover all layers")
    if [item["scope"] for item in sidecar["attention_pattern"]] != LAYER_TYPES:
        raise ValueError("attention pattern is not deterministic")
    if sidecar["model_parameters"]["local_window"] != 512:
        raise ValueError("local window is not preserved")
    if sidecar["aliases"][0]["relation"] != "same_storage":
        raise ValueError("embedding/head alias missing")
    if sidecar["multimodal_boundary"]["selected_path"] != "text_only":
        raise ValueError("selected text boundary changed")


class GemmaSidecarTests(unittest.TestCase):
    def test_pattern_and_optional_multimodal_boundary(self) -> None:
        sidecar = build_sidecar()
        validate(sidecar)
        self.assertEqual([item["layer"] for item in sidecar["attention_pattern"] if item["scope"] == "full_attention"], [5, 11, 17])
        self.assertEqual(sidecar["multimodal_boundary"]["vision_semantics"], "outside_selected_program")

    def test_no_vendor_operation_kind(self) -> None:
        program = json.dumps(build_sidecar()["program"])
        self.assertNotIn("GemmaAttention", program)
        self.assertNotIn("GemmaMLP", program)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "research/results/vbuf-import-boundary-audit/gemma3-typed-program.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[__file__], exit=False).result.wasSuccessful() else 1
    sidecar = build_sidecar()
    validate(sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS — Gemma typed sidecar written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
