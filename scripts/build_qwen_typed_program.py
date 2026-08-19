#!/usr/bin/env python3
"""Build the importer-owned Qwen3.6 P0 typed program sidecar.

The output is deliberately a research/sidecar descriptor, not a vBuf wire
region and not an execution plan.  It records semantic bindings and model
equations while leaving source reads, dispatch results, and scheduling to a
future neutral runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import parse  # noqa: E402


SIDECAR_VERSION = 1
ATTENTION_LAYER = 3
SSM_LAYER = 0
MOE_LAYER = 0


def tensor_map(artifact, layer: int) -> dict[str, object]:
    return {tensor.name.split(".")[-2] if tensor.name.endswith(".weight") else tensor.name.split(".")[-1]: tensor
            for tensor in artifact.tensors if tensor.layer == layer}


def binding(tensor, semantic: str) -> dict[str, object]:
    if tensor.payload_size is None:
        raise ValueError(f"unresolved payload size for {tensor.name}")
    return {
        "semantic": semantic,
        "tensor_ref": {"directory": "TensorDirectory", "semantic_key": semantic},
        "shape": list(tensor.shape),
        "representation": {"kind": "opaque_block_quantized" if tensor.type_name.startswith("Q") else "f32",
                            "block_elements": 256 if tensor.type_name.startswith("Q") else 1,
                            "block_bytes": {"Q4_K": 144, "Q5_K": 176, "Q8_0": 34}.get(tensor.type_name, 4 if tensor.type_name == "F32" else None)},
        "source_provenance": {"name": tensor.name, "source_ordinal": tensor.ordinal, "ggml_type": tensor.type_name,
                               "payload_bytes": tensor.payload_size, "source_offset": tensor.absolute_start},
    }


def require(tensors: dict[str, object], name: str, semantic: str) -> dict[str, object]:
    tensor = tensors.get(name)
    if tensor is None:
        raise ValueError(f"missing required Qwen3.6 tensor: {name}")
    return binding(tensor, semantic)


def build_sidecar(artifact) -> dict[str, Any]:
    metadata = artifact.metadata
    if metadata.get("general.architecture") != "qwen35moe":
        raise ValueError("sidecar requires qwen35moe")
    layer0, layer3 = tensor_map(artifact, SSM_LAYER), tensor_map(artifact, ATTENTION_LAYER)
    moe = tensor_map(artifact, MOE_LAYER)
    bindings = [
        require(layer3, "attn_q", "layer.3.attention.query_weight"),
        require(layer3, "attn_k", "layer.3.attention.key_weight"),
        require(layer3, "attn_v", "layer.3.attention.value_weight"),
        require(layer3, "attn_output", "layer.3.attention.output_weight"),
        require(layer3, "attn_norm", "layer.3.attention.input_norm"),
        require(layer0, "ssm_alpha", "layer.0.ssm.alpha"),
        require(layer0, "ssm_beta", "layer.0.ssm.beta"),
        require(layer0, "ssm_conv1d", "layer.0.ssm.depthwise_conv"),
        require(layer0, "ssm_out", "layer.0.ssm.output_weight"),
        require(moe, "ffn_gate_inp", "layer.0.moe.router_weight"),
        require(moe, "ffn_gate_exps", "layer.0.moe.expert_gate_weight"),
        require(moe, "ffn_up_exps", "layer.0.moe.expert_up_weight"),
        require(moe, "ffn_down_exps", "layer.0.moe.expert_down_weight"),
    ]
    return {
        "sidecar": "vbuf-ml.qwen35moe.typed-program",
        "version": SIDECAR_VERSION,
        "source": {"architecture": "qwen35moe", "artifact": artifact.path.name,
                    "sha256": artifact.sha256, "layer_count": metadata["qwen35moe.block_count"]},
        "portable_boundary": {"excludes": ["ggml_type_ids", "backend_buffers", "device_placement",
                                             "source_read_plan", "current_dispatch", "runtime_schedule"],
                              "tensor_identity": "semantic TensorRef (directory plus occurrence)"},
        "model_parameters": {"embedding": metadata["qwen35moe.embedding_length"],
                             "attention_heads": metadata["qwen35moe.attention.head_count"],
                             "kv_heads": metadata["qwen35moe.attention.head_count_kv"],
                             "head_dimension": metadata["qwen35moe.attention.key_length"],
                             "expert_count": metadata["qwen35moe.expert_count"],
                             "active_experts": metadata["qwen35moe.expert_used_count"]},
        "tensor_bindings": bindings,
        "program": [
            {"id": "attention.layer3", "kind": "attention", "inputs": ["hidden", "position"],
             "outputs": ["attention_output"], "attributes": {"heads": 16, "kv_heads": 2, "head_dimension": 256,
             "causal": True, "position_source": "sequence.position"},
             "weights": ["layer.3.attention.input_norm", "layer.3.attention.query_weight", "layer.3.attention.key_weight",
                          "layer.3.attention.value_weight", "layer.3.attention.output_weight"]},
            {"id": "ssm.layer0", "kind": "ssm_scan", "inputs": ["hidden", "sequence.position"],
             "outputs": ["ssm_output", "ssm_state_next"], "attributes": {"state_size": 128, "group_count": 16,
             "inner_size": 4096, "conv_kernel": 4, "time_step_rank": 32, "equation": "importer_defined_qwen35moe_ssm_v1"},
             "weights": ["layer.0.ssm.alpha", "layer.0.ssm.beta", "layer.0.ssm.depthwise_conv", "layer.0.ssm.output_weight"]},
            {"id": "moe.layer0", "kind": "mixture_of_experts", "inputs": ["hidden"], "outputs": ["moe_output"],
             "attributes": {"expert_count": 256, "top_k": 8, "selection": "router_top_k", "weighting": "router_scores",
                            "capacity": "unspecified", "drop_policy": "unspecified"},
             "router": "layer.0.moe.router_weight", "alternatives": {"selector": "moe.layer0.selected_expert_ids",
                 "set": "layer.0.moe.expert_parameters", "members": ["layer.0.moe.expert_gate_weight",
                 "layer.0.moe.expert_up_weight", "layer.0.moe.expert_down_weight"], "cardinality": 256}}
        ],
        "state_schema": [
            {"ref": "sequence.position", "kind": "position", "shape": [1], "element_type": "u64",
             "transition": "position_next = position + token_count"},
            {"ref": "attention.layer3.kv", "kind": "kv_cache", "shape": [2, 256, 2, "sequence_length"],
             "element_type": "model_storage", "transition": "append_current_key_value"},
            {"ref": "ssm.layer0.state", "kind": "recurrent", "shape": [16, 128, 256],
             "element_type": "f32", "transition": "ssm_scan_state_next"},
        ],
    }


def validate(sidecar: dict[str, Any]) -> None:
    if sidecar["version"] != SIDECAR_VERSION:
        raise ValueError("unsupported sidecar version")
    bindings = sidecar["tensor_bindings"]
    if len({item["semantic"] for item in bindings}) != len(bindings):
        raise ValueError("duplicate semantic binding")
    if len({(item["tensor_ref"]["directory"], item["tensor_ref"]["semantic_key"]) for item in bindings}) != len(bindings):
        raise ValueError("duplicate TensorRef")
    refs = {item["semantic"] for item in bindings}
    for operation in sidecar["program"]:
        for weight in operation.get("weights", []):
            if weight not in refs:
                raise ValueError(f"operation references unbound weight: {weight}")
    moe = next(item for item in sidecar["program"] if item["id"] == "moe.layer0")
    if moe["attributes"]["expert_count"] != moe["alternatives"]["cardinality"]:
        raise ValueError("dynamic alternative cardinality mismatch")


class SidecarTests(unittest.TestCase):
    def test_real_regions_and_portable_exclusions(self) -> None:
        artifact = parse(Path(__file__).resolve().parents[1] / "research-models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        sidecar = build_sidecar(artifact)
        validate(sidecar)
        self.assertEqual(sidecar["program"][0]["id"], "attention.layer3")
        self.assertEqual(sidecar["program"][1]["id"], "ssm.layer0")
        self.assertIn("ggml_type_ids", sidecar["portable_boundary"]["excludes"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=False,
                        default=Path(__file__).resolve().parents[1] / "research-models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    parser.add_argument("--output", type=Path, required=False,
                        default=Path(__file__).resolve().parents[1] / "research/results/vbuf-import-boundary-audit/qwen35moe-typed-program.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1
    sidecar = build_sidecar(parse(args.source.resolve()))
    validate(sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS — typed Qwen3.6 sidecar written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
