#!/usr/bin/env python3
"""Import the qualified DeepSeek-V2-Lite manifest into generic sidecar terms.

This is an import-boundary prototype only. It consumes the checked-in manifest
and does not execute, lower to GGML, or treat GGML representation IDs as
portable semantics.
"""
from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path
from typing import Any


MANIFEST = Path(__file__).resolve().parents[1] / "benchmark-results/vbuf-ml-step18/deepseek-v2-lite-iq1-s-manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plans_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {plan["source_name"]: plan for plan in manifest["tensor_plans"]}


def binding(plan: dict[str, Any], semantic: str) -> dict[str, Any]:
    return {
        "semantic": semantic,
        "tensor_ref": {"directory": "TensorDirectory", "semantic_key": plan["target_name"]},
        "shape": plan["source_shape"],
        "representation": {"kind": "source_representation_contract", "storage": "opaque_or_primitive"},
        "source_provenance": {"name": plan["source_name"], "source_shape": plan["source_shape"],
                               "source_representation": plan["source_ggml_type"]},
    }


def required(plans: dict[str, dict[str, Any]], name: str, semantic: str) -> dict[str, Any]:
    if name not in plans:
        raise ValueError(f"missing DeepSeek source tensor: {name}")
    return binding(plans[name], semantic)


def build_sidecar(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = manifest["source_identity"]
    if identity["architecture"] != "deepseek2":
        raise ValueError("DeepSeek importer requires deepseek2")
    plans = plans_by_name(manifest)
    bindings = [
        required(plans, "token_embd.weight", "embedding.token"),
        required(plans, "output_norm.weight", "output.norm"),
        required(plans, "output.weight", "lm_head.weight"),
        required(plans, "blk.0.attn_norm.weight", "layer.0.attention.input_norm"),
        required(plans, "blk.0.attn_q.weight", "layer.0.attention.query_weight"),
        required(plans, "blk.0.attn_kv_a_mqa.weight", "layer.0.attention.compressed_kv_weight"),
        required(plans, "blk.0.attn_kv_a_norm.weight", "layer.0.attention.compressed_kv_norm"),
        required(plans, "blk.0.attn_kv_b.weight", "layer.0.attention.kv_reconstruction_weight"),
        required(plans, "blk.0.attn_output.weight", "layer.0.attention.output_weight"),
        required(plans, "blk.0.ffn_norm.weight", "layer.0.mlp.input_norm"),
        required(plans, "blk.0.ffn_gate.weight", "layer.0.mlp.gate_weight"),
        required(plans, "blk.0.ffn_up.weight", "layer.0.mlp.up_weight"),
        required(plans, "blk.0.ffn_down.weight", "layer.0.mlp.down_weight"),
        required(plans, "blk.1.ffn_gate_inp.weight", "layer.1.moe.router_weight"),
        required(plans, "blk.1.ffn_gate_shexp.weight", "layer.1.moe.shared_gate_weight"),
        required(plans, "blk.1.ffn_up_shexp.weight", "layer.1.moe.shared_up_weight"),
        required(plans, "blk.1.ffn_down_shexp.weight", "layer.1.moe.shared_down_weight"),
        required(plans, "blk.1.ffn_gate_exps.weight", "layer.1.moe.routed_gate_weight"),
        required(plans, "blk.1.ffn_up_exps.weight", "layer.1.moe.routed_up_weight"),
        required(plans, "blk.1.ffn_down_exps.weight", "layer.1.moe.routed_down_weight"),
    ]
    moe = manifest["moe_plan"]
    return {
        "sidecar": "vbuf-ml.deepseek2.typed-program",
        "version": 1,
        "source": {"architecture": "deepseek2", "artifact": identity["filename"],
                    "sha256": identity["sha256"], "tensor_count": identity["tensor_count"]},
        "portable_boundary": {"excludes": ["ggml_type_ids", "backend_buffers", "device_placement",
                                             "source_read_plan", "runtime_schedule", "current_dispatch"],
                              "tensor_identity": "semantic TensorRef"},
        "model_parameters": {"embedding": 2048, "layers": 27, "heads": 16, "context": 163840,
                             "expert_count": moe["expert_count"], "active_experts": moe["active_expert_count"],
                             "shared_experts": moe["shared_expert_count"]},
        "tensor_bindings": bindings,
        "program": [
            {"id": "embedding", "kind": "embedding", "weights": ["embedding.token"]},
            {"id": "layer.0.attention", "kind": "attention", "weights": ["layer.0.attention.input_norm",
             "layer.0.attention.query_weight", "layer.0.attention.compressed_kv_weight",
             "layer.0.attention.compressed_kv_norm", "layer.0.attention.kv_reconstruction_weight",
             "layer.0.attention.output_weight"], "attributes": {"variant": "compressed_kv_attention",
             "heads": 16, "causal": True, "position_transform": "position.rope"},
             "state": ["attention.kv.compressed", "attention.kv.decoupled_position"]},
            {"id": "layer.1.moe", "kind": "mixture_of_experts", "weights": ["layer.1.moe.router_weight",
             "layer.1.moe.shared_gate_weight", "layer.1.moe.shared_up_weight", "layer.1.moe.shared_down_weight",
             "layer.1.moe.routed_gate_weight", "layer.1.moe.routed_up_weight", "layer.1.moe.routed_down_weight"],
             "attributes": {"expert_count": moe["expert_count"], "top_k": moe["active_expert_count"],
                            "shared_expert_count": moe["shared_expert_count"], "selection": "router_top_k",
                            "weighting": "router_scores"},
             "alternatives": {"selector": "layer.1.moe.selected_expert_ids", "cardinality": moe["expert_count"],
                              "members": ["layer.1.moe.routed_gate_weight", "layer.1.moe.routed_up_weight",
                                           "layer.1.moe.routed_down_weight"]}},
            {"id": "output", "kind": "lm_head", "weights": ["output.norm", "lm_head.weight"]},
        ],
        "state_schema": [
            {"ref": "attention.kv.compressed", "kind": "kv_key", "scope": "layer", "lifetime": "sequence",
             "access": "read-write", "transition": "append(position, compressed_kv); read(position_range)"},
            {"ref": "attention.kv.decoupled_position", "kind": "kv_value", "scope": "layer", "lifetime": "sequence",
             "access": "read-write", "transition": "append(position, positional_kv); read(position_range)"},
            {"ref": "sequence.position", "kind": "position", "scope": "sequence", "lifetime": "sequence",
             "access": "read-write", "transition": "position_next = position + token_count", "reset": "sequence_start"},
        ],
        "legacy_control": {"fixture": "DeepSeek-V2-Lite.IQ1_S", "legacy_poc22_known_correct": True,
                           "comparison_only": True},
    }


def validate(sidecar: dict[str, Any]) -> None:
    bindings = sidecar["tensor_bindings"]
    if len({item["semantic"] for item in bindings}) != len(bindings):
        raise ValueError("duplicate semantic binding")
    refs = {item["semantic"] for item in bindings}
    for operation in sidecar["program"]:
        for weight in operation.get("weights", []):
            if weight not in refs:
                raise ValueError(f"unbound operation weight: {weight}")
    moe = next(operation for operation in sidecar["program"] if operation["id"] == "layer.1.moe")
    if moe["alternatives"]["cardinality"] != sidecar["model_parameters"]["expert_count"]:
        raise ValueError("expert alternative cardinality mismatch")


class DeepSeekSidecarTests(unittest.TestCase):
    def test_manifest_import_is_generic_and_validated(self) -> None:
        sidecar = build_sidecar(load_manifest(MANIFEST))
        validate(sidecar)
        self.assertEqual(sidecar["source"]["tensor_count"], 377)
        self.assertNotIn("GGML", json.dumps(sidecar["portable_boundary"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "research/results/vbuf-import-boundary-audit/deepseek2-typed-program.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[__file__], exit=False).result.wasSuccessful() else 1
    sidecar = build_sidecar(load_manifest(args.manifest))
    validate(sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS — DeepSeek typed sidecar written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
