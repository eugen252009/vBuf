#!/usr/bin/env python3
"""Research-only GGUF inventory and placement qualification for Step 16.

This intentionally is not a GGUF conversion or runtime implementation.  It
extracts bounded descriptor metadata, preserves source payload bytes as an
abstract size, and simulates deterministic placements.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import struct
import sys
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable

EXPECTED = {
    "Q8_0": ("Qwen3-0.6B-Q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"),
    "BF16": ("Qwen3-0.6B-BF16.gguf", "65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e"),
}

# GGML type IDs and exact block geometry needed by the two artifacts.  The
# remaining common IDs are included so a future local artifact fails closed
# only for genuinely unknown geometry.
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    4: ("Q4_2", 32, 18), 5: ("Q4_3", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}
META_TYPES = {0: ("u8", "B"), 1: ("i8", "b"), 2: ("u16", "H"), 3: ("i16", "h"),
              4: ("u32", "I"), 5: ("i32", "i"), 6: ("f32", "f"), 7: ("bool", "?"),
              10: ("u64", "Q"), 11: ("i64", "q"), 12: ("f64", "d")}


def checked_add(a: int, b: int) -> int:
    value = a + b
    if value < a or value > (1 << 64) - 1:
        raise ValueError("u64 addition overflow")
    return value


def checked_mul(a: int, b: int) -> int:
    value = a * b
    if value < 0 or value > (1 << 64) - 1:
        raise ValueError("u64 multiplication overflow")
    return value


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment is not a positive power of two")
    return (value + alignment - 1) // alignment * alignment


class Reader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.pos = 0

    def read(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError(f"truncated GGUF at {self.pos}, need {count} bytes")
        self.pos += count
        return data

    def scalar(self, fmt: str) -> int | float | bool:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        length = int(self.scalar("Q"))
        raw = self.read(length)
        return raw.decode("utf-8", "strict")

    def value(self, value_type: int) -> Any:
        if value_type in META_TYPES:
            return self.scalar(META_TYPES[value_type][1])
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = int(self.scalar("I"))
            count = int(self.scalar("Q"))
            if count > 10_000_000:
                raise ValueError("metadata array is unreasonably large")
            return [self.value(element_type) for _ in range(count)]
        raise ValueError(f"unsupported GGUF metadata type {value_type}")


@dataclass
class Tensor:
    ordinal: int
    name: str
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    elements: int | None
    payload_size: int | None
    relative_offset: int
    absolute_start: int
    absolute_end: int | None
    layer: int | None = None
    role: str = "Unknown"
    group: str = "Unknown"
    role_detail: str = ""
    gap_before: int | None = None
    placement_start: dict[str, int] = field(default_factory=dict)


@dataclass
class Artifact:
    path: Path
    sha256: str
    size: int
    version: int
    tensor_count: int
    metadata_count: int
    metadata: dict[str, Any]
    alignment: int
    data_start: int
    tensors: list[Tensor]
    tensor_info_end: int


def tensor_size(shape: tuple[int, ...], type_id: int) -> tuple[int | None, int | None]:
    elements = 1
    for dimension in shape:
        elements = checked_mul(elements, dimension)
    geometry = TYPES.get(type_id)
    if geometry is None:
        return elements, None
    block, bytes_per_block = geometry[1:]
    if block == 1:
        return elements, checked_mul(elements, bytes_per_block)
    if elements % block:
        return elements, None
    return elements, checked_mul(elements // block, bytes_per_block)


def classify(tensor: Tensor) -> None:
    name = tensor.name
    match = re.search(r"(?:^|\.)(?:blk|block|layers?|h)\.(\d+)(?:\.|$)", name)
    if match:
        tensor.layer = int(match.group(1))
        tensor.group = "Layer"
        tensor.role_detail = name[match.end():]
        tensor.role = tensor.role_detail or "Unknown"
        return
    lower = name.lower()
    if any(token in lower for token in ("token_embd", "tok_embeddings", "embed_tokens", "wte")):
        tensor.group, tensor.role = "GlobalPre", "TokenEmbedding"
    elif any(token in lower for token in ("output_norm", "final_norm", "ln_f")):
        tensor.group, tensor.role = "GlobalPost", "FinalNorm"
    elif lower == "output" or lower.startswith("output.") or lower.endswith(".output") or "lm_head" in lower:
        tensor.group, tensor.role = "GlobalPost", "OutputProjection"
    elif any(token in lower for token in ("rope", "freqs", "attn_scale")):
        tensor.group, tensor.role = "GlobalPre", "OtherGlobal"
    else:
        tensor.group, tensor.role = "Unknown", "Unknown"
    tensor.role_detail = name


def parse(path: Path) -> Artifact:
    size = path.stat().st_size
    with path.open("rb") as stream:
        reader = Reader(stream)
        if reader.read(4) != b"GGUF":
            raise ValueError(f"{path}: invalid GGUF magic")
        version = int(reader.scalar("I"))
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        tensor_count = int(reader.scalar("Q"))
        metadata_count = int(reader.scalar("Q"))
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = reader.string()
            value_type = int(reader.scalar("I"))
            metadata[key] = reader.value(value_type)
        tensors: list[Tensor] = []
        for ordinal in range(tensor_count):
            name = reader.string()
            rank = int(reader.scalar("I"))
            shape = tuple(int(reader.scalar("Q")) for _ in range(rank))
            type_id = int(reader.scalar("I"))
            relative_offset = int(reader.scalar("Q"))
            type_name = TYPES.get(type_id, (f"UNKNOWN_{type_id}", 0, 0))[0]
            elements, payload_size = tensor_size(shape, type_id)
            tensors.append(Tensor(ordinal, name, shape, type_id, type_name, elements, payload_size,
                                  relative_offset, 0, None))
        tensor_info_end = reader.pos
        alignment = int(metadata.get("general.alignment", 32))
        data_start = align_up(tensor_info_end, alignment)
        for tensor in tensors:
            tensor.absolute_start = checked_add(data_start, tensor.relative_offset)
            if tensor.payload_size is not None:
                tensor.absolute_end = checked_add(tensor.absolute_start, tensor.payload_size)
            classify(tensor)
        ordered = sorted(tensors, key=lambda item: (item.absolute_start, item.ordinal))
        previous_end = data_start
        for tensor in ordered:
            tensor.gap_before = tensor.absolute_start - previous_end
            if tensor.absolute_end is not None:
                previous_end = max(previous_end, tensor.absolute_end)
    return Artifact(path, sha256(path), size, version, tensor_count, metadata_count,
                    metadata, alignment, data_start, tensors, tensor_info_end)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layer_count(artifact: Artifact) -> int | None:
    candidates = [artifact.metadata.get(key) for key in ("qwen3.block_count", "llama.block_count", "block_count")]
    values = [int(value) for value in candidates if isinstance(value, int)]
    if values:
        return values[0]
    layers = [tensor.layer for tensor in artifact.tensors if tensor.layer is not None]
    return max(layers) + 1 if layers else None


def metadata_number(artifact: Artifact, *keys: str) -> int | None:
    for key in keys:
        value = artifact.metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def role_key(tensor: Tensor) -> tuple[int, str, int]:
    role = tensor.role_detail.lower()
    ranks = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")
    rank = next((index for index, token in enumerate(ranks) if token in role), len(ranks))
    return rank, role, tensor.ordinal


def layout_order(artifact: Artifact, name: str) -> list[Tensor]:
    source = sorted(artifact.tensors, key=lambda tensor: (tensor.absolute_start, tensor.ordinal))
    if name == "GGUF_SOURCE_ORDER":
        return source
    if name == "NAME_ORDER":
        return sorted(artifact.tensors, key=lambda tensor: tensor.name)
    def key(tensor: Tensor, roles: bool, prefix: bool = False) -> tuple[Any, ...]:
        if tensor.group == "GlobalPre":
            return (0, -1, role_key(tensor) if roles else (tensor.name, tensor.ordinal))
        if tensor.group == "Layer":
            return (1, tensor.layer if tensor.layer is not None else 1 << 60,
                    role_key(tensor) if roles else (tensor.name, tensor.ordinal))
        if tensor.group == "GlobalPost":
            return (3 if prefix else 2, 1 << 60, tensor.name, tensor.ordinal)
        return (4, 1 << 60, tensor.name, tensor.ordinal)
    if name == "LAYER_MAJOR":
        return sorted(artifact.tensors, key=lambda tensor: key(tensor, False))
    if name == "LAYER_MAJOR_ROLE_ORDER":
        return sorted(artifact.tensors, key=lambda tensor: key(tensor, True))
    if name == "PREFIX_FRIENDLY":
        return sorted(artifact.tensors, key=lambda tensor: key(tensor, True, True))
    raise ValueError(f"unknown layout {name}")


LAYOUTS = ("GGUF_SOURCE_ORDER", "NAME_ORDER", "LAYER_MAJOR", "LAYER_MAJOR_ROLE_ORDER", "PREFIX_FRIENDLY")


def simulate(artifact: Artifact, layout: str) -> tuple[list[Tensor], int, int]:
    ordered = layout_order(artifact, layout)
    payload = 0
    padding = 0
    if layout == "GGUF_SOURCE_ORDER":
        for tensor in ordered:
            if tensor.payload_size is None:
                raise ValueError(f"cannot simulate unresolved payload size for {tensor.name} ({tensor.type_name})")
            tensor.placement_start[layout] = tensor.absolute_start
            payload += tensor.payload_size
            padding += tensor.gap_before or 0
        return ordered, payload, padding
    cursor = artifact.data_start
    for tensor in ordered:
        if tensor.payload_size is None:
            raise ValueError(f"cannot simulate unresolved payload size for {tensor.name} ({tensor.type_name})")
        start = align_up(cursor, artifact.alignment)
        padding += start - cursor
        tensor.placement_start[layout] = start
        cursor = start + tensor.payload_size
        payload += tensor.payload_size
    return ordered, payload, padding


def interval(tensor: Tensor, layout: str) -> tuple[int, int]:
    start = tensor.placement_start[layout]
    assert tensor.payload_size is not None
    return start, start + tensor.payload_size


def merge_intervals(tensors: Iterable[Tensor], layout: str, merge_adjacent: bool = True) -> list[tuple[int, int]]:
    ranges = sorted((interval(tensor, layout) for tensor in tensors))
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        joins = bool(merged) and (start <= merged[-1][1] if merge_adjacent else start < merged[-1][1])
        if joins:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def metric(tensors: list[Tensor], artifact: Artifact, layout: str, merge_adjacent: bool = True) -> dict[str, Any]:
    known = [tensor for tensor in tensors if tensor.payload_size is not None]
    semantic = sum(tensor.payload_size or 0 for tensor in known)
    extents = merge_intervals(known, layout, merge_adjacent)
    physical = sum(end - start for start, end in extents)
    first = min((start for start, _ in extents), default=None)
    last = max((end for _, end in extents), default=None)
    span = (last - first) if first is not None and last is not None else 0
    return {"tensor_count": len(tensors), "semantic_bytes": semantic, "physical_bytes": physical,
            "range_count": len(extents), "first_byte": first, "last_byte": last,
            "physical_span": span, "span_amplification": span / semantic if semantic else None,
            "range_amplification": physical / semantic if semantic else None}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_rows(artifact: Artifact, layout: str) -> list[dict[str, Any]]:
    rows = []
    for tensor in sorted(artifact.tensors, key=lambda item: (item.absolute_start, item.ordinal)):
        rows.append({"ordinal": tensor.ordinal, "name": tensor.name, "shape": "x".join(map(str, tensor.shape)),
                     "ggml_type_id": tensor.type_id, "ggml_type": tensor.type_name, "elements": tensor.elements,
                     "payload_bytes": tensor.payload_size, "relative_offset": tensor.relative_offset,
                     "absolute_start": tensor.absolute_start, "absolute_end": tensor.absolute_end,
                     "gap_before": tensor.gap_before, "layer": tensor.layer, "group": tensor.group,
                     "role": tensor.role, "role_detail": tensor.role_detail,
                     "simulated_start": tensor.placement_start.get(layout), "layout": layout})
    return rows


def all_layers(artifact: Artifact) -> list[int]:
    return sorted({tensor.layer for tensor in artifact.tensors if tensor.layer is not None})


def placement_rows(artifact: Artifact, layout: str) -> list[dict[str, Any]]:
    rows = []
    for simulated_ordinal, tensor in enumerate(layout_order(artifact, layout)):
        rows.append({"layout": layout, "simulated_ordinal": simulated_ordinal, "source_ordinal": tensor.ordinal,
                     "name": tensor.name, "layer": tensor.layer, "group": tensor.group, "role": tensor.role,
                     "payload_bytes": tensor.payload_size, "simulated_start": tensor.placement_start[layout],
                     "simulated_end": tensor.placement_start[layout] + (tensor.payload_size or 0)})
    return rows


def prefix_rows(artifact: Artifact, layout: str) -> list[dict[str, Any]]:
    layers = all_layers(artifact)
    pre = [tensor for tensor in artifact.tensors if tensor.group == "GlobalPre"]
    post = [tensor for tensor in artifact.tensors if tensor.group == "GlobalPost"]
    total_payload = sum(tensor.payload_size or 0 for tensor in artifact.tensors)
    full_end = max((interval(tensor, layout)[1] for tensor in artifact.tensors if tensor.payload_size is not None), default=artifact.data_start)
    rows = []
    for last_layer in [-1] + layers:
        selected = pre + [tensor for tensor in artifact.tensors if tensor.layer is not None and tensor.layer <= last_layer]
        none_values = metric(selected, artifact, layout, False)
        exact_values = metric(selected, artifact, layout, True)
        highest = exact_values["last_byte"] or artifact.data_start
        rows.append({"layer_prefix": last_layer, "semantic_bytes": exact_values["semantic_bytes"],
                     "none_physical_read_bytes": none_values["physical_bytes"],
                     "none_physical_range_count": none_values["range_count"],
                     "physical_read_bytes": exact_values["physical_bytes"],
                     "physical_range_count": exact_values["range_count"],
                     "highest_byte_required": highest, "prefix_span_bytes": highest - artifact.data_start,
                     "fraction_model_payload": exact_values["physical_bytes"] / total_payload if total_payload else None,
                     "fraction_model_span": (highest - artifact.data_start) / (full_end - artifact.data_start) if full_end > artifact.data_start else None,
                     "layout": layout})
    return rows


def layer_rows(artifact: Artifact, layout: str) -> list[dict[str, Any]]:
    rows = []
    for layer in all_layers(artifact):
        selected = [tensor for tensor in artifact.tensors if tensor.layer == layer]
        none_values = metric(selected, artifact, layout, False)
        exact_values = metric(selected, artifact, layout, True)
        rows.append({"layer": layer, "tensor_count": exact_values["tensor_count"],
                     "semantic_bytes": exact_values["semantic_bytes"],
                     "none_physical_bytes": none_values["physical_bytes"],
                     "none_range_count": none_values["range_count"],
                     "exact_adjacent_physical_bytes": exact_values["physical_bytes"],
                     "exact_adjacent_range_count": exact_values["range_count"],
                     "first_byte": exact_values["first_byte"], "last_byte": exact_values["last_byte"],
                     "physical_span": exact_values["physical_span"],
                     "span_amplification": exact_values["span_amplification"],
                     "range_amplification": exact_values["range_amplification"], "layout": layout})
    return rows


def distribution_rows(artifact: Artifact) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = {("all", "all"): [tensor.payload_size for tensor in artifact.tensors if tensor.payload_size is not None]}
    for tensor in artifact.tensors:
        if tensor.payload_size is None:
            continue
        groups.setdefault(("layer", str(tensor.layer) if tensor.layer is not None else "unclassified"), []).append(tensor.payload_size)
        groups.setdefault(("role", tensor.role), []).append(tensor.payload_size)
    rows = []
    for (scope, group), values in sorted(groups.items()):
        rows.append({"scope": scope, "group": group, **stats(values)})
    return rows


def type_rows(artifact: Artifact) -> list[dict[str, Any]]:
    totals: dict[tuple[int, str], list[int]] = {}
    for tensor in artifact.tensors:
        key = (tensor.type_id, tensor.type_name)
        entry = totals.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += tensor.payload_size or 0
    total = sum(value[1] for value in totals.values())
    return [{"ggml_type_id": key[0], "ggml_type": key[1], "tensor_count": value[0], "payload_bytes": value[1],
             "payload_fraction": value[1] / total if total else None} for key, value in sorted(totals.items())]


def stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "total": 0}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "median": ordered[(len(ordered) - 1) // 2],
            "p90": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .9) - 1)], "max": ordered[-1], "total": sum(values)}


def qualify(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Artifact] = {}
    for label, (filename, expected_hash) in EXPECTED.items():
        path = root / "research-models" / filename
        if not path.exists():
            print(f"SKIP — research model absent: {path}")
            return
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise SystemExit(f"FAIL — hash mismatch for {path}: expected {expected_hash}, got {actual_hash}")
        artifacts[label] = parse(path)
    for label, artifact in artifacts.items():
        for layout in LAYOUTS:
            simulate(artifact, layout)
        prefix = label.lower().replace("_", "-")
        write_csv(output / f"qwen3-0.6b-{prefix}-tensors.csv", artifact_rows(artifact, "GGUF_SOURCE_ORDER"))
        placements = []
        for layout in LAYOUTS:
            placements.extend(placement_rows(artifact, layout))
        write_csv(output / f"qwen3-0.6b-{prefix}-placements.csv", placements)
        write_csv(output / f"qwen3-0.6b-{prefix}-types.csv", type_rows(artifact))
        write_csv(output / f"qwen3-0.6b-{prefix}-size-distribution.csv", distribution_rows(artifact))
        largest = sorted(artifact.tensors, key=lambda tensor: (tensor.payload_size or -1, tensor.name), reverse=True)[:20]
        write_csv(output / f"qwen3-0.6b-{prefix}-largest-tensors.csv", [{"rank": rank, "name": tensor.name,
                     "payload_bytes": tensor.payload_size, "type": tensor.type_name, "layer": tensor.layer,
                     "group": tensor.group, "role": tensor.role} for rank, tensor in enumerate(largest, 1)])
        layer_rows_all = []
        prefix_rows_all = []
        for layout in LAYOUTS:
            layer_rows_all.extend([{**row, "artifact": label} for row in layer_rows(artifact, layout)])
            prefix_rows_all.extend([{**row, "artifact": label} for row in prefix_rows(artifact, layout)])
        write_csv(output / f"qwen3-0.6b-{prefix}-layer-locality.csv", layer_rows_all)
        write_csv(output / f"qwen3-0.6b-{prefix}-prefix-readiness.csv", prefix_rows_all)
    comparison = []
    for label, artifact in artifacts.items():
        for layout in LAYOUTS:
            ordered, payload, padding = simulate(artifact, layout)
            values = metric(ordered, artifact, layout)
            comparison.append({"artifact": label, "layout": layout, "tensor_count": len(ordered),
                               "payload_bytes": payload, "alignment_padding_bytes": padding,
                               "modeled_tensor_data_span": payload + padding,
                               "range_count_all_tensors": values["range_count"],
                               "range_amplification_all_tensors": values["range_amplification"]})
    write_csv(output / "placement-comparison.csv", comparison)
    names = {label: {tensor.name: tensor for tensor in artifact.tensors} for label, artifact in artifacts.items()}
    inventory = []
    for name in sorted(set(names["Q8_0"]) | set(names["BF16"])):
        left, right = names["Q8_0"].get(name), names["BF16"].get(name)
        inventory.append({"name": name, "q8_present": left is not None, "bf16_present": right is not None,
                          "classification": "same_logical_tensor" if left and right and left.shape == right.shape else
                          "shape_difference" if left and right else "q8_only" if left else "bf16_only",
                          "q8_shape": "x".join(map(str, left.shape)) if left else None,
                          "bf16_shape": "x".join(map(str, right.shape)) if right else None,
                          "q8_type": left.type_name if left else None, "bf16_type": right.type_name if right else None,
                          "q8_group": left.group if left else None, "bf16_group": right.group if right else None})
    write_csv(output / "inventory-comparison.csv", inventory)
    partial = []
    for label, artifact in artifacts.items():
        layers = all_layers(artifact)
        windows = [[layer] for layer in layers[:1]] + [layers[:2], layers[:4], layers[:8], layers]
        if len(layers) > 10:
            windows += [[10], list(range(10, min(14, len(layers))))]
        for layout in LAYOUTS:
            for window in windows:
                selected = [tensor for tensor in artifact.tensors if tensor.layer in window]
                none_values = metric(selected, artifact, layout, False)
                exact_values = metric(selected, artifact, layout, True)
                partial.append({"artifact": label, "layout": layout, "selection": f"layers:{window[0]}-{window[-1]}",
                                "semantic_bytes": exact_values["semantic_bytes"],
                                "none_physical_bytes": none_values["physical_bytes"], "none_range_count": none_values["range_count"],
                                "exact_adjacent_physical_bytes": exact_values["physical_bytes"], "exact_adjacent_range_count": exact_values["range_count"],
                                "span_amplification": exact_values["span_amplification"]})
    write_csv(output / "partial-loading-comparison.csv", partial)
    boundaries = []
    for label, artifact in artifacts.items():
        for layout in LAYOUTS:
            for layer in all_layers(artifact):
                tensors = [tensor for tensor in artifact.tensors if tensor.layer == layer]
                ends = [interval(tensor, layout)[1] for tensor in tensors]
                boundaries.append({"artifact": label, "layout": layout, "boundary": f"after_layer_{layer}",
                                   "layer": layer, "cumulative_end": max(ends) if ends else None,
                                   "offset_from_data_start": max(ends) - artifact.data_start if ends else None})
    write_csv(output / "natural-boundaries.csv", boundaries)
    index_rows = []
    for label, artifact in artifacts.items():
        start = time.perf_counter()
        index = {layer: [tensor.ordinal for tensor in artifact.tensors if tensor.layer == layer] for layer in all_layers(artifact)}
        elapsed = time.perf_counter() - start
        approximate_bytes = sys.getsizeof(index) + sum(sys.getsizeof(key) + sys.getsizeof(value) + sum(sys.getsizeof(item) for item in value) for key, value in index.items())
        lookups = max(1, len(index) * 1000)
        lookup_start = time.perf_counter()
        for _ in range(lookups):
            _ = index.get(_ % max(1, len(index)))
        lookup_ns = (time.perf_counter() - lookup_start) * 1e9 / lookups
        index_rows.append({"artifact": label, "layer_count": len(index), "construction_seconds": elapsed,
                           "approximate_memory_bytes": approximate_bytes, "lookup_count": lookups,
                           "lookup_nanoseconds": lookup_ns})
    write_csv(output / "layer-index.csv", index_rows)
    summaries = []
    for label, artifact in artifacts.items():
        summaries.append({"artifact": label, "filename": artifact.path.name, "file_size": artifact.size,
                          "sha256": sha256(artifact.path), "gguf_version": artifact.version,
                          "metadata_kv_count": artifact.metadata_count, "tensor_count": artifact.tensor_count,
                          "data_start": artifact.data_start, "alignment": artifact.alignment,
                          "architecture": artifact.metadata.get("general.architecture"),
                          "name": artifact.metadata.get("general.name"), "layer_count": layer_count(artifact),
                          "embedding_dimension": metadata_number(artifact, "qwen3.embedding_length", "llama.embedding_length", "embedding_length"),
                          "attention_heads": metadata_number(artifact, "qwen3.attention.head_count", "llama.attention.head_count"),
                          "kv_heads": metadata_number(artifact, "qwen3.attention.head_count_kv", "llama.attention.head_count_kv"),
                          "vocabulary_size": metadata_number(artifact, "qwen3.vocab_size", "llama.vocab_size") or (len(artifact.metadata["tokenizer.ggml.tokens"]) if isinstance(artifact.metadata.get("tokenizer.ggml.tokens"), list) else None)})
    write_csv(output / "artifact-summary.csv", summaries)
    metadata_keys = []
    for label, artifact in artifacts.items():
        for key, value in sorted(artifact.metadata.items()):
            if isinstance(value, list):
                rendered = f"array[{len(value)}]"
            elif isinstance(value, (str, int, float, bool)):
                rendered = str(value)
            else:
                rendered = type(value).__name__
            metadata_keys.append({"artifact": label, "key": key, "value_type": type(value).__name__, "value_or_shape": rendered})
    write_csv(output / "metadata-key-inventory.csv", metadata_keys)
    (output / "qualification-config.json").write_text(json.dumps({"repository_commit": git_commit(root), "python": sys.version,
        "platform": platform.platform(), "machine": platform.machine(), "layouts": LAYOUTS,
        "alignment_policy": "artifact general.alignment, default 32; payloads unchanged", "parser": "scripts/qualify_step16.py"}, indent=2) + "\n", encoding="utf-8")
    print(f"PASS — Step 16 qualification complete for {', '.join(artifacts)}; evidence: {output}")


def git_commit(root: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class ToolTests(unittest.TestCase):
    def test_checked_layout_and_type_size(self) -> None:
        self.assertEqual(tensor_size((32, 4), 8), (128, 136))
        self.assertEqual(align_up(33, 32), 64)
        self.assertEqual(tensor_size((31,), 8), (31, None))

    def test_classification_unknown_is_explicit(self) -> None:
        tensor = Tensor(0, "future.tensor", (1,), 0, "F32", 1, 4, 0, 0, 4)
        classify(tensor)
        self.assertEqual((tensor.group, tensor.role), ("Unknown", "Unknown"))

    def test_simulations_preserve_payload_bytes(self) -> None:
        artifact = Artifact(Path("x"), "", 0, 3, 2, 0, {}, 32, 0, [], 0)
        artifact.tensors = [Tensor(0, "blk.0.a", (1,), 0, "F32", 1, 4, 0, 0, 4),
                            Tensor(1, "blk.1.b", (2,), 0, "F32", 2, 8, 32, 32, 40)]
        for tensor in artifact.tensors:
            classify(tensor)
        baseline = sum(tensor.payload_size for tensor in artifact.tensors)
        for layout in LAYOUTS:
            self.assertEqual(simulate(artifact, layout)[1], baseline)

    def test_deterministic_role_order(self) -> None:
        artifact = Artifact(Path("x"), "", 0, 3, 2, 0, {}, 32, 0, [], 0)
        a = Tensor(0, "blk.0.ffn_down.weight", (1,), 0, "F32", 1, 4, 0, 0, 4)
        b = Tensor(1, "blk.0.attn_norm.weight", (1,), 0, "F32", 1, 4, 32, 32, 36)
        for tensor in (a, b):
            classify(tensor)
        artifact.tensors = [a, b]
        self.assertEqual([tensor.name for tensor in layout_order(artifact, "LAYER_MAJOR_ROLE_ORDER")], [b.name, a.name])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.main(argv=[sys.argv[0]], exit=False).result.wasSuccessful() else 1
    output = args.output_dir or args.root / "benchmark-results" / "vbuf-ml-step16"
    try:
        qualify(args.root.resolve(), output.resolve())
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL — {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
