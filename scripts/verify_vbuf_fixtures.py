#!/usr/bin/env python3
"""Verify immutable legacy-v0.5 byte and behavior evidence.

The manifest records observed implementation behavior. It does not normalize
legacy defects or define the corrected v0.6 contract.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "v05"
MANIFEST = FIXTURES / "manifest.json"


def fail(message: str) -> None:
    raise SystemExit(f"legacy fixture verification failed: {message}")


def checked_file(entry: dict[str, object]) -> bytes:
    path = FIXTURES / str(entry["path"])
    data = path.read_bytes()
    if len(data) != entry["size"]:
        fail(f"{entry['path']}: size mismatch")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        fail(f"{entry['path']}: SHA-256 mismatch")
    return data


def verify_valid_fixture(entry: dict[str, object], data: bytes, shared: dict[str, object]) -> None:
    if data[:4].hex() != shared["magic_bytes_hex"]:
        fail(f"{entry['path']}: magic mismatch")
    if struct.unpack_from("<I", data, 4)[0] != shared["version_u32_le"]:
        fail(f"{entry['path']}: version mismatch")
    if data[8] != shared["stored_a_shift"]:
        fail(f"{entry['path']}: stored AShift mismatch")
    if struct.unpack_from("<I", data, 12)[0] != entry["data_len_u32_le"]:
        fail(f"{entry['path']}: DataLen mismatch")

    for column, block_offset in zip(entry["columns"], (16, 48), strict=True):
        anchor = struct.unpack_from("<Q", data, block_offset)[0]
        actual = {
            "semantic": anchor & 0xF,
            "key_id": (anchor >> 16) & 0xFFFF,
            "bit_width": (anchor >> 32) & 0xFFFF,
            "count": struct.unpack_from("<Q", data, block_offset + 8)[0],
        }
        if not anchor & (1 << 9):
            fail(f"{entry['path']}: expected overflow-count encoding")
        for field, value in actual.items():
            if value != column[field]:
                fail(f"{entry['path']}: {field} mismatch at block {block_offset}")
        payload_offset = (block_offset + 16 + 15) & ~15
        if payload_offset != column["payload_offset"]:
            fail(f"{entry['path']}: payload offset mismatch")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {entry["path"] for entry in manifest["files"]}
    present = {p.name for p in FIXTURES.iterdir() if p.is_file() and p.name != "manifest.json"}
    if listed != present:
        fail(f"manifest/files differ: listed={sorted(listed)}, present={sorted(present)}")

    entries = {entry["path"]: entry for entry in manifest["files"]}
    bytes_by_name: dict[str, bytes] = {}
    for entry in manifest["files"]:
        data = checked_file(entry)
        bytes_by_name[entry["path"]] = data
        if "columns" in entry:
            verify_valid_fixture(entry, data, manifest["shared_observations"])
        else:
            if "observed_legacy_behavior" not in entry or "expected_v06" not in entry:
                fail(f"{entry['path']}: malformed evidence lacks behavior/v0.6 annotation")

    original = bytes_by_name["current-typescript.vbuf"]
    truncated = bytes_by_name["truncated-typescript.vbuf"]
    if truncated != original[:-8]:
        fail("truncated-typescript.vbuf is not the one-Float64 truncation")
    anchor = struct.unpack_from("<Q", truncated, 48)[0]
    count = struct.unpack_from("<Q", truncated, 56)[0]
    if ((anchor >> 32) & 0xFFFF, count, len(truncated) - 64) != (64, 2, 8):
        fail("truncated fixture no longer declares 2 x Float64 with one available")
    if len(bytes_by_name["short.vbuf"]) >= 16:
        fail("short.vbuf is not short")
    if bytes_by_name["bad-magic.vbuf"][:4] == b"VBUF":
        fail("bad-magic.vbuf unexpectedly has VBUF magic")

    print(f"validated {len(entries)} immutable legacy-v0.5 byte/behavior fixtures")


if __name__ == "__main__":
    main()
