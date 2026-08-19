#!/usr/bin/env python3
"""Step-17 research qualification for the pinned F32/BF16/Q8_0 contracts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import EXPECTED, parse  # noqa: E402

PINNED_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
PINNED_COMMIT = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"

PROVENANCE_FILES = {
    "ggml/include/ggml.h": ["GGML_TYPE_F32", "GGML_TYPE_Q8_0", "GGML_TYPE_BF16"],
    "ggml/src/ggml-common.h": ["#define QK8_0 32", "typedef struct {", "block_q8_0"],
    "ggml/src/ggml.c": ["[GGML_TYPE_Q8_0]", "[GGML_TYPE_BF16]", "size_t ggml_row_size"],
    "ggml/src/ggml-impl.h": ["ggml_compute_bf16_to_fp32", "ggml_compute_fp32_to_bf16"],
    "ggml/src/ggml-quants.c": ["quantize_row_q8_0_ref", "dequantize_row_q8_0"],
    "src/models/qwen3.cpp": ["output = create_tensor", "TENSOR_DUPLICATED"],
    "src/llama-model-loader.cpp": ["some models use the token embedding tensor as the output", "TENSOR_DUPLICATED"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_mul(a: int, b: int) -> int:
    result = a * b
    if result > (1 << 64) - 1:
        raise ValueError("u64 multiplication overflow")
    return result


def expected_bytes(type_name: str, shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements = checked_mul(elements, dimension)
    if type_name == "F32":
        return checked_mul(elements, 4)
    if type_name == "BF16":
        return checked_mul(elements, 2)
    if type_name == "Q8_0":
        if not shape or shape[0] % 32:
            raise ValueError("Q8_0 innermost row is not divisible by 32")
        rows = 1
        for dimension in shape[1:]:
            rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 32), 34)
    if type_name == "Q4_0":
        if not shape or shape[0] % 32:
            raise ValueError("Q4_0 innermost row is not divisible by 32")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 32), 18)
    if type_name == "Q2_K":
        if not shape or shape[0] % 256:
            raise ValueError("Q2_K innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 84)
    if type_name == "IQ1_S":
        if not shape or shape[0] % 256:
            raise ValueError("IQ1_S innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 50)
    if type_name in {"Q4_K", "Q4_K_S", "Q4_K_M"}:
        if not shape or shape[0] % 256:
            raise ValueError("Q4_K innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 144)
    if type_name in {"Q3_K", "Q3_K_S", "Q3_K_M"}:
        if not shape or shape[0] % 256:
            raise ValueError("Q3_K innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 110)
    if type_name == "Q5_K":
        if not shape or shape[0] % 256:
            raise ValueError("Q5_K innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 176)
    if type_name == "Q6_K":
        if not shape or shape[0] % 256:
            raise ValueError("Q6_K innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 210)
    if type_name == "IQ2_XXS":
        block_bytes = 66
    elif type_name == "IQ2_XS":
        block_bytes = 74
    elif type_name == "IQ2_S":
        block_bytes = 84
    else:
        block_bytes = None
    if block_bytes is not None:
        if not shape or shape[0] % 256:
            raise ValueError(f"{type_name} innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), block_bytes)
    if type_name == "IQ4_NL":
        if not shape or shape[0] % 32:
            raise ValueError("IQ4_NL innermost row is not divisible by 32")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 32), 18)
    if type_name == "IQ4_XS":
        if not shape or shape[0] % 256:
            raise ValueError("IQ4_XS innermost row is not divisible by 256")
        rows = 1
        for dimension in shape[1:]: rows = checked_mul(rows, dimension)
        return checked_mul(checked_mul(rows, shape[0] // 256), 136)
    raise ValueError(f"unsupported Step-17 type {type_name}")


def source_provenance(root: Path) -> dict[str, object]:
    files: dict[str, object] = {}
    for relative, needles in PROVENANCE_FILES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"pinned source file is absent: {relative}")
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = []
        for needle in needles:
            match = next(((index + 1, line.strip()) for index, line in enumerate(lines) if needle in line), None)
            if match is None:
                raise ValueError(f"pinned source symbol/text is absent: {relative}: {needle}")
            matches.append({"needle": needle, "line": match[0], "text": match[1]})
        files[relative] = matches
    return files


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-root", type=Path, default=Path("/tmp/llama.cpp-step17"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = args.upstream_root.resolve()
    output = (args.output_dir or root / "benchmark-results" / "vbuf-ml-step17").resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not upstream.exists():
        print(f"SKIP — pinned upstream checkout absent: {upstream}")
        return 0
    try:
        actual_commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL — upstream checkout is not a git repository: {error}", file=sys.stderr)
        return 1
    if actual_commit != PINNED_COMMIT:
        print(f"FAIL — upstream commit mismatch: expected {PINNED_COMMIT}, got {actual_commit}", file=sys.stderr)
        return 1
    try:
        provenance = source_provenance(upstream)
    except (OSError, ValueError) as error:
        print(f"FAIL — upstream provenance: {error}", file=sys.stderr)
        return 1

    artifacts = {}
    for label, (filename, expected_hash) in EXPECTED.items():
        path = root / "research-models" / filename
        if not path.exists():
            print(f"SKIP — research model absent: {path}")
            return 0
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            print(f"FAIL — hash mismatch for {path}: expected {expected_hash}, got {actual_hash}", file=sys.stderr)
            return 1
        try:
            artifacts[label] = parse(path)
        except (OSError, ValueError) as error:
            print(f"FAIL — GGUF parse: {error}", file=sys.stderr)
            return 1

    rows = []
    for label, artifact in artifacts.items():
        for tensor in artifact.tensors:
            try:
                expected = expected_bytes(tensor.type_name, tensor.shape)
                error = ""
            except ValueError as failure:
                expected = None
                error = str(failure)
            actual = tensor.payload_size
            rows.append({"artifact": label, "name": tensor.name, "shape": "x".join(map(str, tensor.shape)),
                         "ggml_type_id": tensor.type_id, "ggml_type": tensor.type_name,
                         "actual_payload_bytes": actual, "expected_payload_bytes": expected,
                         "match": expected == actual, "error": error})
    failures = [row for row in rows if not row["match"]]
    write_csv(output / "representation-parity.csv", rows)
    write_csv(output / "q8_0-shape-parity.csv", [row for row in rows if row["ggml_type"] == "Q8_0"])
    write_csv(output / "bf16-parity.csv", [row for row in rows if row["ggml_type"] == "BF16"])
    write_csv(output / "f32-parity.csv", [row for row in rows if row["ggml_type"] == "F32"])
    (output / "upstream-provenance.json").write_text(json.dumps({
        "repository": PINNED_REPOSITORY,
        "commit": PINNED_COMMIT,
        "source_provenance": provenance,
        "contracts": {
            "F32": {"ggml_type_id": 0, "bytes_per_element": 4},
            "BF16": {"ggml_type_id": 30, "bytes_per_element": 2},
            "Q8_0": {"ggml_type_id": 8, "block_elements": 32, "block_bytes": 34, "row_rule": "innermost dimension divisible by 32"},
        },
        "output_fallback": "src/models/qwen3.cpp load_arch_tensors creates output.weight as not required and duplicates token_embd.weight with TENSOR_DUPLICATED when absent",
    }, indent=2) + "\n", encoding="utf-8")
    (output / "qualification-config.json").write_text(json.dumps({
        "repository_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_commit": PINNED_COMMIT, "python": sys.version, "platform": platform.platform(),
        "artifacts": {label: {"filename": artifact.path.name, "size": artifact.size, "sha256": sha256(artifact.path)} for label, artifact in artifacts.items()},
    }, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"FAIL — {len(failures)} representation geometry mismatches; evidence: {output}", file=sys.stderr)
        return 1
    print(f"PASS — F32/BF16/Q8_0 parity for {len(rows)} tensors; evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
