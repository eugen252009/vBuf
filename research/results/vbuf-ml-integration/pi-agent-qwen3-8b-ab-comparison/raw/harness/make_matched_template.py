#!/usr/bin/env python3
"""Derive a llama.cpp Qwen3 template matching vBuf's native prompt rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_GGUF_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
OLD_TOOL_SERIALIZER = "{{- tool | tojson }}"
NEW_TOOL_SERIALIZER = r'''{{- tool | tojson | replace("'", "\\u0027") }}'''
OLD_THINKING_SUFFIX = r'''    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-python-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.gguf_python_path))
    from gguf import GGUFReader

    reader = GGUFReader(str(args.gguf), "r")
    field = reader.fields.get("tokenizer.chat_template")
    if field is None:
        raise SystemExit("GGUF lacks tokenizer.chat_template")
    original = field.parts[-1].tobytes().decode("utf-8")
    original_bytes = original.encode("utf-8")
    if sha256_file(args.gguf) != EXPECTED_GGUF_SHA256:
        raise SystemExit("GGUF identity mismatch; refusing to derive the matched template")
    if original.count(OLD_TOOL_SERIALIZER) != 1:
        raise SystemExit("expected exactly one Qwen tool-schema serializer expression")
    if original.count(OLD_THINKING_SUFFIX) != 1:
        raise SystemExit("expected exactly one no-thinking generation suffix block")

    matched = original.replace(OLD_TOOL_SERIALIZER, NEW_TOOL_SERIALIZER, 1)
    matched = matched.replace(OLD_THINKING_SUFFIX, "", 1)
    output_bytes = matched.encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output_bytes)
    metadata = {
        "schema": "qwen3-vbuf-matched-chat-template-v1",
        "gguf_sha256": EXPECTED_GGUF_SHA256,
        "original_template_sha256": sha256(original_bytes),
        "matched_template_sha256": sha256(output_bytes),
        "changes": [
            "Escape apostrophes in serialized tool schemas as \\u0027, matching serialize_qwen_template_json.",
            "Omit llama.cpp's empty <think>...</think> generation suffix, matching the vBuf native renderer's assistant prefix.",
        ],
        "source_template_chars": len(original),
        "matched_template_chars": len(matched),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
