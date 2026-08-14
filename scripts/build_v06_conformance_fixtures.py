#!/usr/bin/env python3
"""Build or verify deterministic cross-language vBuf v0.6 conformance fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "v06"
VERSION = 0x00060000


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def anchor(*, semantic=0, physical=1, continuation=False, count64=False,
           payload_shift=0, key=1, width=32, count=1, inline_override=None) -> int:
    inline = (0 if count64 else count) if inline_override is None else inline_override
    return (semantic | (physical << 4) | (int(continuation) << 8) |
            (int(count64) << 9) | (payload_shift << 10) | (key << 16) |
            (width << 32) | (inline << 48))


def global_header(base_shift: int, data_size: int, *, flags=0,
                  header_size=24, extension=b"", reserved=0) -> bytearray:
    raw = bytearray(b"VBUF" + struct.pack("<I", VERSION))
    raw += bytes((base_shift, flags)) + struct.pack("<H", header_size)
    raw += struct.pack("<I", reserved) + struct.pack("<Q", data_size)
    raw += extension
    assert len(raw) == header_size
    raw += bytes(align_up(header_size, 1 << base_shift) - header_size)
    return raw


def canonical(blocks: list[dict], *, base_shift=3, flags=0,
              header_size=24, extension=b"") -> bytearray:
    step = 1 << base_shift
    data_start = align_up(header_size, step)
    data = bytearray()
    for index, block in enumerate(blocks):
        block_start = data_start + len(data)
        assert block_start % step == 0
        count = block["count"]
        count64 = block.get("count64", count > 65535)
        raw_anchor = anchor(
            semantic=block.get("semantic", 0), physical=block.get("physical", 1),
            continuation=block.get("continuation", False), count64=count64,
            payload_shift=block.get("payload_shift", 0), key=block.get("key", 1),
            width=block.get("width", 8), count=count,
        )
        data += struct.pack("<Q", raw_anchor)
        if count64:
            data += struct.pack("<Q", count)
        payload_alignment = step << block.get("payload_shift", 0)
        payload_start = align_up(data_start + len(data), payload_alignment)
        data += bytes(payload_start - (data_start + len(data)))
        data += block.get("payload", b"")
        if index + 1 < len(blocks):
            next_start = align_up(data_start + len(data), step)
            data += bytes(next_start - (data_start + len(data)))
    size = 0 if flags & 1 else len(data)
    return global_header(base_shift, size, flags=flags, header_size=header_size,
                         extension=extension) + data


def set_u16(raw: bytearray, offset: int, value: int) -> bytearray:
    raw[offset:offset + 2] = struct.pack("<H", value)
    return raw


def set_u32(raw: bytearray, offset: int, value: int) -> bytearray:
    raw[offset:offset + 4] = struct.pack("<I", value)
    return raw


def set_u64(raw: bytearray, offset: int, value: int) -> bytearray:
    raw[offset:offset + 8] = struct.pack("<Q", value)
    return raw


def fixtures() -> dict[str, tuple[bool, bytearray, str]]:
    basic = canonical([{"key": 1, "width": 32, "count": 3,
                        "payload": struct.pack("<III", 1, 2, 3)}])
    float_truncated_data = struct.pack("<d", 1.5)
    optional = struct.pack("<HHI", 99, 0, 8)
    out: dict[str, tuple[bool, bytearray, str]] = {
        "valid-basic.vbuf": (True, basic, "known-size u32 array; partial final BaseStep"),
        "valid-empty.vbuf": (True, global_header(3, 0), "known-size empty stream"),
        "valid-zero-array.vbuf": (True, canonical([{"key": 2, "width": 8, "count": 0, "payload": b""}]), "zero-length array"),
        "valid-duplicate-chain.vbuf": (True, canonical([
            {"key": 7, "width": 8, "count": 1, "payload": b"A", "continuation": True},
            {"key": 7, "width": 8, "count": 1, "payload": b"B"},
        ]), "duplicate KeyID with valid forward continuation"),
        "valid-duplicate-unchained.vbuf": (True, canonical([
            {"key": 7, "width": 8, "count": 1, "payload": b"A"},
            {"key": 7, "width": 8, "count": 1, "payload": b"B"},
        ]), "duplicate KeyID without grouping"),
        "valid-optional-extension.vbuf": (True, canonical([], header_size=32, extension=optional), "unknown optional extension skipped"),
        "valid-indefinite.vbuf": (True, canonical([{"key": 1, "width": 32, "count": 3, "payload": struct.pack("<III", 1, 2, 3)}], flags=1), "complete indefinite stream at EOF"),
        "short-header.vbuf": (False, bytearray(b"VBUF\x00\x00\x06\x00"), "short global header"),
        "bad-magic.vbuf": (False, bytearray(basic[:]), "bad magic"),
        "unsupported-version.vbuf": (False, bytearray(basic[:]), "unsupported version"),
        "base-shift-low.vbuf": (False, bytearray(basic[:]), "BaseShift below range"),
        "base-shift-high.vbuf": (False, bytearray(basic[:]), "BaseShift above range"),
        "unknown-flags.vbuf": (False, bytearray(basic[:]), "unknown global flag"),
        "invalid-header-size.vbuf": (False, bytearray(basic[:]), "HeaderSize not divisible by eight"),
        "reserved-nonzero.vbuf": (False, bytearray(basic[:]), "reserved field non-zero"),
        "nonzero-global-padding.vbuf": (False, canonical([], base_shift=4), "non-zero global alignment padding"),
        "truncated-anchor.vbuf": (False, global_header(3, 4) + b"\x10\x00\x00\x00", "truncated block anchor"),
        "truncated-extended-count.vbuf": (False, global_header(3, 8) + struct.pack("<Q", anchor(count64=True, width=8, count=70000)), "truncated extended count"),
        "count-overflow.vbuf": (False, global_header(3, 16) + struct.pack("<QQ", anchor(count64=True, width=64, count=0), (1 << 64) - 1), "count times width overflow"),
        "payload-truncated-byte.vbuf": (False, canonical([{"key": 1, "width": 32, "count": 3, "payload": struct.pack("<III", 1, 2, 3)}])[:-1], "payload truncated by one byte"),
        "payload-truncated-element.vbuf": (False, canonical([{"key": 2, "semantic": 1, "width": 64, "count": 2, "payload": float_truncated_data}]), "two Float64 declared; one available"),
        "invalid-payload-shift.vbuf": (False, global_header(3, 8) + struct.pack("<Q", anchor(payload_shift=61, width=8, count=1)), "combined payload shift exceeds 63"),
        "invalid-semantic.vbuf": (False, bytearray(basic[:]), "reserved semantic"),
        "invalid-physical.vbuf": (False, bytearray(basic[:]), "reserved physical mode"),
        "unsupported-width.vbuf": (False, bytearray(basic[:]), "unsupported integer width"),
        "impossible-scalar-count.vbuf": (False, global_header(3, 8) + struct.pack("<Q", anchor(physical=0, width=8, count=0)), "scalar count is not one"),
        "malformed-extension-zero.vbuf": (False, global_header(3, 0, header_size=32, extension=struct.pack("<HHI", 99, 0, 0)), "zero extension length"),
        "malformed-extension-overrun.vbuf": (False, global_header(3, 0, header_size=32, extension=struct.pack("<HHI", 99, 0, 16)), "extension exceeds HeaderSize"),
        "unknown-required-extension.vbuf": (False, global_header(3, 0, header_size=32, extension=struct.pack("<HHI", 99, 1, 8)), "unknown required extension"),
        "known-size-mismatch.vbuf": (False, bytearray(basic[:]), "known DataRegionSize differs from physical length"),
        "known-size-overflow.vbuf": (False, bytearray(basic[:24]), "data-region end addition overflow"),
        "known-size-over-4g.vbuf": (False, bytearray(basic[:24]), "declared range exceeds 4 GiB and physical input"),
        "indefinite-nonzero-size.vbuf": (False, bytearray(basic[:]), "indefinite stream carries size"),
        "indefinite-truncated.vbuf": (False, canonical([{"key": 2, "semantic": 1, "width": 64, "count": 2, "payload": float_truncated_data}], flags=1), "truncated final indefinite block"),
        "final-tail-padding.vbuf": (False, bytearray(basic + b"\x00\x00\x00\x00"), "unnecessary final padding"),
        "nonzero-payload-padding.vbuf": (False, canonical([{"key": 1, "width": 32, "count": 1, "payload": struct.pack("<I", 1)}], base_shift=4), "non-zero payload alignment padding"),
        "unaligned-block-start.vbuf": (False, canonical([
            {"key": 1, "width": 8, "count": 1, "payload": b"A"},
            {"key": 2, "width": 8, "count": 1, "payload": b"B"},
        ]), "non-zero bytes before canonical next block"),
        "noncanonical-extended-small.vbuf": (False, global_header(3, 16) + struct.pack("<QQ", anchor(count64=True, width=8, count=1), 1), "small count uses extended form"),
        "noncanonical-extended-inline.vbuf": (False, global_header(3, 16) + struct.pack("<QQ", anchor(count64=True, width=8, count=70000, inline_override=1), 70000), "extended form has inline count"),
        "continuation-final.vbuf": (False, canonical([{"key": 7, "width": 8, "count": 1, "payload": b"A", "continuation": True}]), "final block sets Continuation"),
        "chain-key-mismatch.vbuf": (False, canonical([
            {"key": 7, "width": 8, "count": 1, "payload": b"A", "continuation": True},
            {"key": 8, "width": 8, "count": 1, "payload": b"B"},
        ]), "forward continuation KeyID differs from next block"),
    }
    out["bad-magic.vbuf"][1][0] = 0
    set_u32(out["unsupported-version.vbuf"][1], 4, 0x00050000)
    out["base-shift-low.vbuf"][1][8] = 2
    out["base-shift-high.vbuf"][1][8] = 9
    out["unknown-flags.vbuf"][1][9] = 2
    set_u16(out["invalid-header-size.vbuf"][1], 10, 25)
    set_u32(out["reserved-nonzero.vbuf"][1], 12, 1)
    out["nonzero-global-padding.vbuf"][1][24] = 1
    # Keep the physical bytes truncated while making DataRegionSize exact.
    set_u64(out["payload-truncated-byte.vbuf"][1], 16, len(out["payload-truncated-byte.vbuf"][1]) - 24)
    set_u64(out["payload-truncated-element.vbuf"][1], 16, len(out["payload-truncated-element.vbuf"][1]) - 24)
    data_start = 24
    raw_anchor = struct.unpack_from("<Q", out["invalid-semantic.vbuf"][1], data_start)[0]
    struct.pack_into("<Q", out["invalid-semantic.vbuf"][1], data_start, (raw_anchor & ~0xf) | 4)
    raw_anchor = struct.unpack_from("<Q", out["invalid-physical.vbuf"][1], data_start)[0]
    struct.pack_into("<Q", out["invalid-physical.vbuf"][1], data_start, (raw_anchor & ~(0xf << 4)) | (2 << 4))
    raw_anchor = struct.unpack_from("<Q", out["unsupported-width.vbuf"][1], data_start)[0]
    struct.pack_into("<Q", out["unsupported-width.vbuf"][1], data_start, (raw_anchor & ~(0xffff << 32)) | (24 << 32))
    set_u64(out["known-size-mismatch.vbuf"][1], 16, 21)
    set_u64(out["known-size-overflow.vbuf"][1], 16, (1 << 64) - 1)
    set_u64(out["known-size-over-4g.vbuf"][1], 16, 1 << 32)
    out["indefinite-nonzero-size.vbuf"][1][9] = 1
    set_u64(out["final-tail-padding.vbuf"][1], 16, 24)
    out["nonzero-payload-padding.vbuf"][1][40] = 1
    # First payload ends at 33; canonical second block starts at 40.
    out["unaligned-block-start.vbuf"][1][34] = 0x10
    return out


def manifest_for(items: dict[str, tuple[bool, bytearray, str]]) -> dict:
    return {
        "schema": "vbuf-v06-cross-language-conformance-1",
        "authority": "spec/spec_0.6.md",
        "files": [
            {"path": name, "accept": accept, "size": len(raw),
             "sha256": hashlib.sha256(raw).hexdigest(), "case": case}
            for name, (accept, raw, case) in sorted(items.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    items = fixtures()
    manifest = manifest_for(items)
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, (_, raw, _) in items.items():
            (OUT / name).write_bytes(raw)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {len(items)} v0.6 conformance fixtures")
        return 0
    actual = json.loads((OUT / "manifest.json").read_text())
    if actual != manifest:
        raise SystemExit("v0.6 fixture manifest differs from deterministic source")
    for name, (_, raw, _) in items.items():
        if (OUT / name).read_bytes() != raw:
            raise SystemExit(f"v0.6 fixture differs: {name}")
    print(f"verified {len(items)} v0.6 cross-language conformance fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
