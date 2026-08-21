#!/usr/bin/env python3
"""Offline Step 31N expert-bank rank/provenance audit.

This consumes the already-qualified import manifest only. It does not modify
the manifest, vBuf artifacts, runtime metadata, or backend execution path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKS = {
    "IQ2_XXS": (256, 66),
    "IQ4_NL": (32, 18),
}
ROLES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def expected_payload_bytes(type_name: str, shape: list[int]) -> int:
    block_elements, block_bytes = BLOCKS[type_name]
    if len(shape) != 3 or shape[0] == 0 or shape[0] % block_elements != 0:
        raise ValueError(f"invalid {type_name} bank shape: {shape}")
    rows = shape[1] * shape[2]
    return rows * (shape[0] // block_elements) * block_bytes


def bank_records(plan: dict[str, Any]) -> list[dict[str, int]]:
    shape = plan["source_shape"]
    stride = plan["source_payload_bytes"] // shape[2]
    return [
        {
            "expert": expert,
            "offset": plan["source_offset"] + expert * stride,
            "length": stride,
        }
        for expert in range(shape[2])
    ]


def audit_bank(plan: dict[str, Any], all_plans: list[dict[str, Any]]) -> dict[str, Any]:
    shape = plan["source_shape"]
    payload = plan["source_payload_bytes"]
    if len(shape) != 3:
        raise ValueError(f"{plan['source_name']} is not rank 3")
    expert_count = shape[2]
    if expert_count == 0 or payload % expert_count != 0:
        raise ValueError(f"{plan['source_name']} has invalid expert division")

    expected = expected_payload_bytes(plan["source_ggml_type"], shape)
    if expected != payload:
        raise ValueError(
            f"{plan['source_name']} payload mismatch: expected {expected}, got {payload}"
        )

    records = bank_records(plan)
    stride = records[0]["length"]
    adjacent = [records[index + 1]["offset"] - records[index]["offset"]
                for index in range(expert_count - 1)]
    if any(value != stride for value in adjacent):
        raise ValueError(f"{plan['source_name']} expert stride is not fixed")
    if records[-1]["offset"] + stride != plan["source_offset"] + payload:
        raise ValueError(f"{plan['source_name']} expert records do not cover bank span")

    bank_start = plan["source_offset"]
    bank_end = bank_start + payload
    for candidate in all_plans:
        if candidate is plan:
            continue
        candidate_start = candidate["source_offset"]
        candidate_end = candidate_start + candidate["source_payload_bytes"]
        if candidate_start < bank_end and bank_start < candidate_end:
            raise ValueError(
                f"{plan['source_name']} overlaps {candidate['source_name']}"
            )

    reconstructed = {
        "rank": 3,
        "dims": [shape[0], shape[1], expert_count],
        "base": min(record["offset"] for record in records),
        "stride": stride,
        "span": stride * expert_count,
        "dtype": plan["source_ggml_type"],
        "offsets": [record["offset"] for record in records],
    }
    original = {
        "rank": len(shape),
        "dims": shape,
        "base": plan["source_offset"],
        "stride": stride,
        "span": payload,
        "dtype": plan["source_ggml_type"],
        "offsets": [plan["source_offset"] + index * stride for index in range(expert_count)],
    }
    if reconstructed != original:
        raise ValueError(f"{plan['source_name']} rank round-trip mismatch")

    return {
        "name": plan["source_name"],
        "role": plan["source_name"].split(".", 2)[2],
        "layer": plan["layer"],
        "rank": len(shape),
        "dims": shape,
        "expert_count": expert_count,
        "per_expert_dims": shape[:2],
        "dtype": plan["source_ggml_type"],
        "base": plan["source_offset"],
        "stride": stride,
        "payload": payload,
        "span": reconstructed["span"],
        "offsets_match": reconstructed["offsets"] == original["offsets"],
        "order_match": [record["expert"] for record in bank_records(plan)]
        == list(range(expert_count)),
    }


def run(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plans = manifest["tensor_plans"]
    banks = [plan for plan in plans if any(plan["source_name"].endswith(role) for role in ROLES)]
    if len(banks) != 78:
        raise ValueError(f"expected 78 routed expert banks, found {len(banks)}")

    results = [audit_bank(plan, plans) for plan in banks]
    representative = [
        result for result in results
        if result["name"] in {
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
        }
    ]
    print("STEP31N_RANK_DERIVATION: PASS")
    print(f"BANKS_CHECKED: {len(results)}")
    print(f"LAYERS_CHECKED: {len({result['layer'] for result in results})}")
    print("CURRENT_RANK2_RECORDS: EPHEMERAL_SELECTED_VIEWS_NOT_PERSISTED")
    for result in representative:
        print(
            "BANK name={name} rank={rank} dims={dims} expert_count={expert_count} "
            "per_expert_dims={per_expert_dims} dtype={dtype} base={base} "
            "stride={stride} payload={payload} span={span} offsets_match={offsets_match} "
            "order_match={order_match}".format(**result)
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        return run(args.manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"STEP31N_RANK_DERIVATION: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
