#!/usr/bin/env python3
"""Verify normative arithmetic and byte vectors for generic vBuf v0.6."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "spec" / "vectors" / "v06-vectors.json"
U64_MAX = (1 << 64) - 1


class VectorError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def checked_add(a: int, b: int) -> int:
    value = a + b
    if a < 0 or b < 0 or value > U64_MAX:
        raise VectorError("u64_overflow")
    return value


def checked_mul(a: int, b: int) -> int:
    value = a * b
    if a < 0 or b < 0 or value > U64_MAX:
        raise VectorError("u64_overflow")
    return value


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise VectorError("invalid_alignment")
    return checked_add(value, alignment - 1) & ~(alignment - 1)


def base_step(base_shift: int) -> int:
    if not 3 <= base_shift <= 8:
        raise VectorError("invalid_base_shift")
    return 1 << base_shift


def data_region_start(base_shift_value: int, header_size: int) -> int:
    if header_size < 24 or header_size % 8:
        raise VectorError("invalid_header_size")
    return align_up(header_size, base_step(base_shift_value))


def payload_alignment(base_shift_value: int, payload_shift: int) -> int:
    combined = base_shift_value + payload_shift
    if payload_shift < 0 or combined > 63:
        raise VectorError("invalid_payload_shift")
    return 1 << combined


def payload_bytes(count: int, bit_width: int) -> int:
    bits = checked_mul(count, bit_width)
    return bits // 8 + (1 if bits % 8 else 0)


def validate_count_form(count64: bool, count: int, inline_count: int) -> None:
    if count64:
        if count <= 65535 or inline_count != 0:
            raise VectorError("noncanonical_count")
    elif count > 65535 or inline_count != count:
        raise VectorError("noncanonical_count")


def verify_valid_geometry(vectors: dict) -> int:
    checked = 0
    for vector in vectors["valid_base_geometry"]:
        step = base_step(vector["base_shift"])
        start = data_region_start(vector["base_shift"], vector["header_size"])
        assert step == vector["base_step"], vector
        assert start == vector["data_region_start"], vector
        assert start % step == 0, vector
        checked += 1
    return checked


def verify_block_offsets(vectors: dict) -> int:
    checked = 0
    for vector in vectors["block_offset_cases"]:
        step = base_step(vector["base_shift"])
        for case in vector["cases"]:
            offset = case["offset"]
            assert 0 <= offset <= U64_MAX, case
            assert (offset % step == 0) == case["representable"], (vector, case)
            checked += 1
    return checked


def verify_valid_blocks(vectors: dict) -> int:
    checked = 0
    for vector in vectors["valid_blocks"]:
        expected = vector["expected"]
        step = base_step(vector["base_shift"])
        start = vector["block_start"]
        assert start % step == 0, vector
        header_size = 16 if vector["count64"] else 8
        validate_count_form(
            vector["count64"],
            vector["count"],
            0 if vector["count64"] else vector["count"],
        )
        alignment = payload_alignment(vector["base_shift"], vector["payload_shift"])
        payload_start = align_up(checked_add(start, header_size), alignment)
        byte_count = payload_bytes(vector["count"], vector["bit_width"])
        payload_end = checked_add(payload_start, byte_count)
        next_start = align_up(payload_end, step)
        actual = {
            "header_size": header_size,
            "payload_alignment": alignment,
            "payload_start": payload_start,
            "payload_bytes": byte_count,
            "payload_end": payload_end,
            "next_block_start": next_start,
            "final_data_region_size": payload_end - min(
                item["data_region_start"]
                for item in vectors["valid_base_geometry"]
                if item["base_shift"] == vector["base_shift"] and item["header_size"] == 24
            ),
        }
        assert actual == expected, (vector["name"], actual, expected)
        assert payload_start % step == 0, vector
        assert payload_start % alignment == 0, vector
        checked += 1
    return checked


def verify_file_hex(vectors: dict) -> int:
    checked = 0
    for vector in vectors["canonical_file_hex"]:
        raw = bytes.fromhex(vector["hex"])
        assert len(raw) == vector["file_size"], vector
        assert raw[:4] == b"VBUF", vector
        assert struct.unpack_from("<I", raw, 4)[0] == 0x00060000, vector
        shift = raw[8]
        assert base_step(shift) == vector["base_step"], vector
        header_size = struct.unpack_from("<H", raw, 10)[0]
        start = data_region_start(shift, header_size)
        size = struct.unpack_from("<Q", raw, 16)[0]
        assert start == vector["data_region_start"], vector
        assert size == vector["data_region_size"], vector
        assert checked_add(start, size) == len(raw), vector
        anchor = struct.unpack_from("<Q", raw, start)[0]
        count = (anchor >> 48) & 0xFFFF
        width = (anchor >> 32) & 0xFFFF
        pstart = align_up(start + 8, base_step(shift))
        pend = pstart + payload_bytes(count, width)
        assert pstart == vector["payload_start"] and pend == len(raw), vector
        values = list(struct.unpack_from(f"<{count}I", raw, pstart))
        assert values == vector["payload_values_u32"], vector
        checked += 1
    return checked


def run_invalid(vector: dict) -> None:
    operation = vector["operation"]
    if operation == "base_step":
        base_step(vector["base_shift"])
    elif operation == "data_region_start":
        data_region_start(vector["base_shift"], vector["header_size"])
    elif operation == "payload_alignment":
        payload_alignment(vector["base_shift"], vector["payload_shift"])
    elif operation == "payload_bytes":
        payload_bytes(vector["count"], vector["bit_width"])
    elif operation == "align_up":
        align_up(vector["value"], vector["alignment"])
    elif operation == "count_form":
        validate_count_form(vector["count64"], vector["count"], vector["inline_count"])
    elif operation == "known_data_end":
        checked_add(vector["data_region_start"], vector["data_region_size"])
    else:
        raise AssertionError(f"unknown operation: {operation}")


def verify_invalid(vectors: dict) -> int:
    checked = 0
    for vector in vectors["invalid"]:
        try:
            run_invalid(vector)
        except VectorError as error:
            assert error.code == vector["error"], (vector, error.code)
        else:
            raise AssertionError(f"invalid vector accepted: {vector['name']}")
        checked += 1
    return checked


def main() -> int:
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert vectors["schema"] == "vbuf-v06-normative-vectors-1"
    total = sum(
        (
            verify_valid_geometry(vectors),
            verify_block_offsets(vectors),
            verify_valid_blocks(vectors),
            verify_file_hex(vectors),
            verify_invalid(vectors),
        )
    )
    print(f"verified {total} normative vBuf v0.6 vectors")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        print(f"v0.6 vector verification failed: {error}", file=sys.stderr)
        sys.exit(1)
