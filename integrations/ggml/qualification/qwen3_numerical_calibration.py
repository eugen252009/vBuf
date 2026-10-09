#!/usr/bin/env python3
"""Provenance-bound replay/calibration for captured Qwen3-14B CUDA operators.

The script reads immutable model captures, replays saved Q/K through the pinned
GGML CUDA graph at selected physical KV capacities, and stores diagnostic-only
high-precision comparisons. It never edits the external GGML checkout; optional
kernel-route instrumentation is applied temporarily to the generated build copy
and restored before exit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL_SHA = "f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31"
GGUF_SHA = "915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6"
GGML_COMMIT = "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"
PLACEMENT = "multi:0x26,1x14;emb=0;norm=1;head=1"
CAPACITIES = (256, 512, 1024, 1032)
CUDA_REPLAY_CAPACITIES = (512, 1024, 1032)
D, QH, KVH = 128, 40, 8


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], *, cwd: Path | None = None,
                env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def parse_key_value_meta(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def capture_layout(data: bytes, meta: dict[str, str]) -> dict[str, tuple[int, int]]:
    rows, visible = int(meta["query_rows"]), int(meta["visible_context"])
    values = D * visible * KVH
    probabilities = visible * rows * QH
    outputs = D * rows * QH
    scores = visible * rows * QH
    cursor = 0
    layout: dict[str, tuple[int, int]] = {}
    for name, count, width in (
        ("V_F16", values, 2), ("P_F32", probabilities, 4),
        ("positions_I32", rows, 4), ("canonical_F32", outputs, 4),
        ("native_F32", outputs, 4), ("oracle_F32", outputs, 4),
        ("K_F16", values, 2), ("Q_F32", D * rows * QH, 4),
        ("scores_F32", scores, 4), ("qk_oracle_F32", scores, 4),
        ("softmax_from_scores_F32", probabilities, 4),
        ("softmax_from_qk_oracle_F32", probabilities, 4),
    ):
        length = count * width
        layout[name] = (cursor, length)
        cursor += length
    if len(data) != cursor:
        raise ValueError(f"capture byte length mismatch: got {len(data)}, expected {cursor}")
    return layout


def logical_score_bytes(data: bytes, meta: dict[str, str], layout: dict[str, tuple[int, int]]) -> bytes:
    rows, visible = int(meta["query_rows"]), int(meta["visible_context"])
    start, _ = layout["scores_F32"]
    values = __import__("struct").unpack_from(f"<{rows * QH * visible}f", data, start)
    logical = [0.0] * len(values)
    for row in range(rows):
        for head in range(QH):
            for position in range(visible):
                logical[(row * QH + head) * visible + position] = values[
                    position + visible * (row + rows * head)]
    return __import__("struct").pack(f"<{len(logical)}f", *logical)


def capture_identity(directory: Path, capacity: int, phase: str) -> dict[str, Any]:
    stem = f"boundary-capacity-{capacity}-{phase}-layer-00"
    meta_path, binary_path = directory / f"{stem}.meta", directory / f"{stem}.bin"
    meta = parse_key_value_meta(meta_path)
    expected = {
        "format": "qwen3-native-av-boundary-v1", "capacity": str(capacity),
        "layer": "0", "device_id": "0", "phase": phase,
        "query_heads": str(QH), "kv_heads": str(KVH), "head_dim": str(D),
        "qk_capture": "yes",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{meta_path}: {key}={meta.get(key)!r}, expected {value!r}")
    data = binary_path.read_bytes()
    layout = capture_layout(data, meta)
    k_start, k_size = layout["K_F16"]
    q_start, q_size = layout["Q_F32"]
    return {
        "meta_path": str(meta_path), "meta_sha256": sha256_file(meta_path),
        "binary_path": str(binary_path), "binary_sha256": sha256_bytes(data),
        "capacity": capacity, "phase": phase, "rows": int(meta["query_rows"]),
        "visible_context": int(meta["visible_context"]),
        "key_input_sha256": sha256_bytes(data[k_start:k_start + k_size]),
        "query_input_sha256": sha256_bytes(data[q_start:q_start + q_size]),
        "saved_cuda_qk_sha256": sha256_bytes(logical_score_bytes(data, meta, layout)),
        "_meta": meta, "_data": data, "_layout": layout,
    }


def source_dispatch(capacity: int, rows: int, cc: int) -> str:
    small = capacity <= 512
    if (cc >= 80 and small and rows == 1) or (70 <= cc < 80 and small and rows <= 3):
        return "MMVF"
    if cc >= 70 and D % 64 == 0 and capacity % 32 == 0 and rows <= 16:
        return "MMF"
    return "cuBLAS"


def parse_eval_records(stdout: str) -> list[dict[str, Any]]:
    records = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and ("contract_id" in record or "record_type" in record):
            records.append(record)
    return records


def instrumented_cuda_source(original: str) -> str:
    start = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {\n    GGML_TENSOR_BINARY_OP_LOCALS\n"
    trace = '''\n    const bool trace_qwen_calibration = std::getenv("VBUF_QWEN_CALIBRATION_TRACE_DISPATCH") != nullptr &&
        src0->type == GGML_TYPE_F16 && src0->ne[0] == 128 && src0->ne[2] == 8 &&
        src1->type == GGML_TYPE_F32 && src1->ne[0] == 128 && src1->ne[2] == 40 &&
        dst->type == GGML_TYPE_F32 && dst->ne[2] == 40;
    auto trace_qwen_route = [&](const char * route) {
        if (trace_qwen_calibration) {
            std::fprintf(stderr, "QWEN_QK_DISPATCH route=%s device=%d cc=%d K=[%lld,%lld,%lld] Q=[%lld,%lld,%lld] scores=[%lld,%lld,%lld]\\n",
                route, ctx.device, ggml_cuda_info().devices[ctx.device].cc,
                (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2],
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2]);
        }
    };
'''
    if original.count(start) != 1:
        raise ValueError("pinned ggml-cuda.cu did not match the expected mul_mat dispatch function")
    result = original.replace(start, start + trace, 1)
    replacements = (
        ("if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {\n        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);",
         "if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {\n        trace_qwen_route(\"CUBLAS_PRECHECK\");\n        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);"),
        ("if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {\n        //",
         "if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {\n        trace_qwen_route(\"MMVF\");\n        //"),
        ("if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {\n        ggml_cuda_mul_mat_f",
         "if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {\n        trace_qwen_route(\"MMF\");\n        ggml_cuda_mul_mat_f"),
        ("    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);\n}\n\n// returns true when ggml_cuda_mul_mat_id",
         "    trace_qwen_route(\"CUBLAS_FINAL\");\n    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);\n}\n\n// returns true when ggml_cuda_mul_mat_id"),
    )
    for old, new in replacements:
        if result.count(old) != 1:
            raise ValueError(f"could not uniquely instrument dispatch branch: {old[:90]!r}")
        result = result.replace(old, new, 1)
    return result


def extract_rmsnorm_weight(gguf_path: Path, gguf_python_path: Path, output: Path) -> str:
    sys.path.insert(0, str(gguf_python_path))
    try:
        gguf = importlib.import_module("gguf")
        import numpy as np
    except Exception as error:
        raise RuntimeError(f"could not load gguf-py/numpy for exact norm weight extraction: {error}") from error
    reader = gguf.GGUFReader(str(gguf_path), "r")
    tensor = next((item for item in reader.tensors if item.name == "blk.21.attn_norm.weight"), None)
    if tensor is None or tuple(tensor.shape) != (5120,) or tensor.data.dtype != np.dtype("float32"):
        raise ValueError("source GGUF blk.21.attn_norm.weight is not F32[5120]")
    np.asarray(tensor.data, dtype="<f4").tofile(output)
    if output.stat().st_size != 5120 * 4:
        raise ValueError("extracted RMSNorm weight has an unexpected size")
    return sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/vbuf-qwen-native-matrix-build"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capture-root", type=Path,
        default=ROOT / "research/results/vbuf-ml-integration/qwen3-native-layout-av/raw/av-boundary-qk")
    parser.add_argument("--repeat-root", type=Path,
        default=ROOT / "research/results/vbuf-ml-integration/qwen3-native-layout-av/raw/av-boundary-qk-repeat")
    parser.add_argument("--operator-capture", type=Path,
        default=Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/resident-canonical-prefix32-layer21-position24-v3/operator-capture"))
    parser.add_argument("--operator-run-log", type=Path,
        default=Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/resident-canonical-prefix32-layer21-position24-v3/run.log"))
    parser.add_argument("--operator-build-cache", type=Path,
        default=Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/build-residency/CMakeCache.txt"))
    parser.add_argument("--model-gguf", type=Path,
        default=Path("/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.gguf"))
    parser.add_argument("--gguf-python-path", type=Path, default=Path("/home/eugen/projekte/llama.cpp/gguf-py"))
    parser.add_argument("--ggml-source", type=Path, default=Path("/home/eugen/.cache/vbuf-agent-qualification/ggml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    build = args.build_dir.resolve()
    exe_capture = build / "vbuf_qwen3_reference_capture_qualification"
    exe_cuda = build / "vbuf_qwen3_cuda_qk_calibration"
    if not exe_capture.is_file() or not exe_cuda.is_file():
        raise RuntimeError("build the CUDA qualification targets first; expected executables in " + str(build))
    cache_lines = (build / "CMakeCache.txt").read_text().splitlines()
    cache = {}
    for line in cache_lines:
        if line.startswith("VBUF_ENABLE_CUDA:") or line.startswith("VBUF_GGML_COMMIT:") or \
                line.startswith("VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS:"):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for key, expected in (("VBUF_ENABLE_CUDA", "ON"), ("VBUF_GGML_COMMIT", GGML_COMMIT),
                          ("VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS", "OFF")):
        if cache.get(key) != expected:
            raise RuntimeError(f"CMake cache {key} must be {expected}, got {cache.get(key)!r}")

    external_commit = run_checked(["git", "-C", str(args.ggml_source), "rev-parse", "HEAD"]).stdout.strip()
    if external_commit != GGML_COMMIT:
        raise RuntimeError(f"external GGML source commit mismatch: {external_commit}")
    source_file = args.ggml_source / "src/ggml-cuda/ggml-cuda.cu"
    external_cuda_sha = sha256_file(source_file)
    external_status = run_checked(["git", "-C", str(args.ggml_source), "status", "--short"]).stdout
    external_diff = run_checked(["git", "-C", str(args.ggml_source), "diff", "--", "src/ggml-cuda/ggml-cuda.cu"]).stdout
    if not args.model_gguf.is_file():
        raise RuntimeError("source Qwen3 GGUF is missing")
    model_gguf_sha = sha256_file(args.model_gguf)
    if model_gguf_sha != GGUF_SHA:
        raise RuntimeError("source Qwen3 GGUF does not match its registered SHA-256")

    source_logs = {}
    run_roots = {"primary": args.capture_root.resolve(), "repeat": args.repeat_root.resolve()}
    for run_name, directory in run_roots.items():
        log_path = directory / "capacity-sweep.log"
        log_text = log_path.read_text()
        if MODEL_SHA not in log_text:
            raise RuntimeError(f"{log_path} is not bound to the expected vBuf model SHA")
        source_logs[run_name] = {"path": str(log_path), "sha256": sha256_file(log_path)}

    captures: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for run_name, directory in run_roots.items():
        captures[run_name] = {}
        for capacity in CAPACITIES:
            captures[run_name][capacity] = {}
            for phase in ("prefill", "decode"):
                identity = capture_identity(directory, capacity, phase)
                del identity["_meta"], identity["_data"], identity["_layout"]
                captures[run_name][capacity][phase] = identity
        for phase in ("prefill", "decode"):
            q_hashes = {captures[run_name][c][phase]["query_input_sha256"] for c in CAPACITIES}
            k_hashes = {captures[run_name][c][phase]["key_input_sha256"] for c in CAPACITIES}
            if len(q_hashes) != 1 or len(k_hashes) != 1:
                raise RuntimeError(f"Q/K are not byte-identical across {run_name} capacity runs ({phase})")
    for capacity in CAPACITIES:
        for phase in ("prefill", "decode"):
            for field in ("query_input_sha256", "key_input_sha256"):
                if captures["primary"][capacity][phase][field] != captures["repeat"][capacity][phase][field]:
                    raise RuntimeError(f"repeat capture changed {field}: capacity={capacity} phase={phase}")

    operator_meta = args.operator_capture / "resident-capture.meta"
    operator_text = operator_meta.read_text()
    if "layer=21 position=24 positions=32" not in operator_text:
        raise RuntimeError("unexpected layer-21 operator capture geometry")
    if MODEL_SHA not in args.operator_run_log.read_text():
        raise RuntimeError("layer-21 operator run does not identify the expected model SHA")
    operator_cache = args.operator_build_cache.read_text()
    if f"VBUF_GGML_COMMIT:STRING={GGML_COMMIT}" not in operator_cache:
        raise RuntimeError("layer-21 operator capture build is not bound to the pinned GGML commit")

    input_files = [Path(source_logs[name]["path"]) for name in source_logs]
    input_files.extend(Path(value[key]) for runs in captures.values() for capacities in runs.values()
                       for value in capacities.values() for key in ("meta_path", "binary_path"))
    input_files += [operator_meta, args.operator_run_log, args.operator_build_cache, args.model_gguf,
                    source_file, build / "CMakeCache.txt",
                    build / "vbuf-patched-ggml/src/ggml-cuda/ggml-cuda.cu",
                    Path(__file__).resolve()]
    input_inventory = []
    for path in sorted(set(p.resolve() for p in input_files)):
        input_inventory.append({"path": str(path), "size_bytes": path.stat().st_size,
                                "sha256": sha256_file(path)})

    default_output = ROOT / "research/results/vbuf-ml-integration/qwen3-cuda-numerical-calibration/raw" / (
        "run-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {
        "schema_version": 1, "model_identity": f"sha256:{MODEL_SHA}",
        "source_gguf_sha256": GGUF_SHA, "ggml_commit": GGML_COMMIT,
        "placement_identity": PLACEMENT, "policy_id": "vbuf.numerical-contracts",
        "policy_version": 2, "high_precision_reference_revision": "vbuf-high-precision-reference-cpp-v1",
        "build_configuration": {"directory": str(build), "cuda": cache["VBUF_ENABLE_CUDA"],
            "unvalidated_qualification_trials": cache["VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS"],
            "ggml_commit": cache["VBUF_GGML_COMMIT"],
            "cmake_cache_sha256": sha256_file(build / "CMakeCache.txt")},
        "external_ggml_worktree_status": external_status,
        "external_ggml_cuda_source_sha256": external_cuda_sha,
        "external_ggml_existing_diff_sha256": sha256_bytes(external_diff.encode()),
        "external_ggml_existing_diff": external_diff,
        "source_logs": source_logs, "captures": captures,
        "operator_capture": {"directory": str(args.operator_capture.resolve()),
            "metadata_sha256": sha256_file(operator_meta), "run_log_sha256": sha256_file(args.operator_run_log),
            "build_cache_sha256": sha256_file(args.operator_build_cache)},
        "inputs": input_inventory, "capture_replays": [], "cuda_qk_replays": [],
        "calibration_authority": "diagnostic-only; no contract thresholds or admission scope activated",
    }
    (output_dir / "input-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.validate_only:
        print(f"capture inputs validated; provenance={output_dir / 'input-provenance.json'}")
        return 0

    all_records: list[dict[str, Any]] = []
    for capacity in CAPACITIES:
        destination = output_dir / "historical-capture-replays" / f"capacity-{capacity}.log"
        command = [str(exe_capture), str(args.capture_root.resolve()), str(capacity)]
        result = run_checked(command)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout + result.stderr)
        records = parse_eval_records(result.stdout)
        if not records:
            raise RuntimeError(f"capture replay produced no Numerical Contract records for capacity {capacity}")
        all_records.extend(records)
        provenance["capture_replays"].append({"capacity": capacity, "source_run": "primary",
            "command": command, "log": str(destination), "log_sha256": sha256_file(destination),
            "evaluation_records": len(records)})

    with tempfile.TemporaryDirectory(prefix="qwen3-norm-weight-", dir=output_dir) as temporary:
        weight_path = Path(temporary) / "blk.21.attn_norm.weight.f32"
        weight_sha = extract_rmsnorm_weight(args.model_gguf, args.gguf_python_path, weight_path)
        command = [str(exe_capture), "--operator-capture", str(args.operator_capture.resolve()),
                   str(weight_path), weight_sha, GGUF_SHA]
        result = run_checked(command)
        destination = output_dir / "layer21-operator-capture.log"
        destination.write_text(result.stdout + result.stderr)
        op_records = parse_eval_records(result.stdout)
        if len(op_records) < 4:
            raise RuntimeError("layer-21 operator replay did not produce QK/softmax/RMSNorm/AV records")
        all_records.extend(op_records)
        provenance["operator_capture"]["norm_weight_tensor"] = "blk.21.attn_norm.weight:F32[5120]"
        provenance["operator_capture"]["norm_weight_tensor_sha256"] = weight_sha
        provenance["operator_capture"]["log"] = str(destination)
        provenance["operator_capture"]["log_sha256"] = sha256_file(destination)
        provenance["operator_capture"]["evaluation_records"] = len(op_records)

    generated_source = build / "vbuf-patched-ggml/src/ggml-cuda/ggml-cuda.cu"
    original_source = generated_source.read_text()
    traced_source = instrumented_cuda_source(original_source)
    patch = "".join(difflib.unified_diff(original_source.splitlines(True), traced_source.splitlines(True),
                                         fromfile="a/src/ggml-cuda/ggml-cuda.cu",
                                         tofile="b/src/ggml-cuda/ggml-cuda.cu"))
    (output_dir / "cuda-dispatch-trace.patch").unlink(missing_ok=True)
    (output_dir / "cuda-dispatch-trace.patch.gz").write_bytes(
        gzip.compress(patch.encode(), mtime=0))
    trace_build_log = output_dir / "cuda-dispatch-instrumented-build.log"
    restore_build_log = output_dir / "cuda-dispatch-restored-build.log"
    try:
        generated_source.write_text(traced_source)
        result = subprocess.run(["cmake", "--build", str(build), "--target",
                                 "vbuf_qwen3_cuda_qk_calibration", "-j2"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        trace_build_log.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError("diagnostic dispatch-instrumented target build failed; see " + str(trace_build_log))
        for phase in ("prefill", "decode"):
            for capacity in CUDA_REPLAY_CAPACITIES:
                for device in (0, 1):
                    expected_route = source_dispatch(capacity, 32 if phase == "prefill" else 1,
                                                     86 if device == 0 else 75)
                    output = output_dir / "cuda-qk" / f"cuda{device}" / phase / f"capacity-{capacity}.f32"
                    case_log = output.with_suffix(".log")
                    command = [str(exe_cuda), str(args.capture_root.resolve()), "512",
                               str(capacity), phase, str(device), str(output)]
                    env = os.environ.copy()
                    env["VBUF_QWEN_CALIBRATION_TRACE_DISPATCH"] = "1"
                    result = run_checked(command, env=env)
                    stderr = result.stderr
                    raw_routes = re.findall(r"QWEN_QK_DISPATCH route=([A-Z_]+)", stderr)
                    routes = ["cuBLAS" if route.startswith("CUBLAS") else route for route in raw_routes]
                    if routes != [expected_route]:
                        raise RuntimeError(f"observed CUDA route {raw_routes} does not match pinned-source route "
                                           f"{expected_route}: {case_log}")
                    case_log.parent.mkdir(parents=True, exist_ok=True)
                    case_log.write_text(result.stdout + result.stderr)
                    records = parse_eval_records(result.stdout)
                    if len(records) != 1:
                        raise RuntimeError(f"expected one QK evaluation record, found {len(records)}")
                    all_records.extend(records)
                    source_capture = capture_identity(args.capture_root.resolve(), capacity, phase)
                    source_data = (args.capture_root.resolve() /
                        f"boundary-capacity-{capacity}-{phase}-layer-00.bin").read_bytes()
                    source_meta = parse_key_value_meta(args.capture_root.resolve() /
                        f"boundary-capacity-{capacity}-{phase}-layer-00.meta")
                    source_layout = capture_layout(source_data, source_meta)
                    captured = logical_score_bytes(source_data, source_meta, source_layout)
                    replayed = output.read_bytes()
                    if len(captured) != len(replayed):
                        raise RuntimeError("replayed QK output length differs from its full-model capture")
                    import struct
                    expected_values = struct.unpack(f"<{len(captured) // 4}f", captured)
                    actual_values = struct.unpack(f"<{len(replayed) // 4}f", replayed)
                    differences = [float(a) - float(b) for a, b in zip(actual_values, expected_values)]
                    abs_errors = [abs(value) for value in differences]
                    rms_error = (sum(value * value for value in differences) / len(differences)) ** 0.5
                    ref_rms = (sum(float(value) * float(value) for value in expected_values) / len(expected_values)) ** 0.5
                    match_record = {
                        "record_type": "replay_vs_full_model_capture",
                        "model_identity": f"sha256:{MODEL_SHA}", "phase": phase,
                        "source_capacity": 512, "execution_capacity": capacity,
                        "device": device, "device_sm": 86 if device == 0 else 75,
                        "dispatch_family": expected_route,
                        "logical_inputs_equivalent": True,
                        "source_capture_query_sha256": source_capture["query_input_sha256"],
                        "source_capture_key_sha256": source_capture["key_input_sha256"],
                        "full_model_capture_score_sha256": sha256_bytes(captured),
                        "replayed_score_sha256": sha256_bytes(replayed),
                        "bitwise_equal": captured == replayed,
                        "max_absolute_error": max(abs_errors), "rms_error": rms_error,
                        "relative_rms_error": rms_error / ref_rms if ref_rms else None,
                    }
                    match_path = output.with_suffix(".comparison.json")
                    match_path.write_text(json.dumps(match_record, indent=2) + "\n")
                    provenance["cuda_qk_replays"].append({
                        "phase": phase, "capacity": capacity, "device": device,
                        "dispatch_family": expected_route, "output": str(output),
                        "output_sha256": sha256_file(output), "log": str(case_log),
                        "log_sha256": sha256_file(case_log), "comparison": str(match_path),
                        "comparison_sha256": sha256_file(match_path),
                        "evaluation_record": records[0], "full_capture_match": match_record,
                    })
    finally:
        generated_source.write_text(original_source)
        result = subprocess.run(["cmake", "--build", str(build), "--target",
                                 "vbuf_qwen3_cuda_qk_calibration", "-j2"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        restore_build_log.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError("failed to restore the uninstrumented CUDA build; see " + str(restore_build_log))

    # Preserve the raw contract evaluation records exactly as emitted by C++.
    evaluations_path = output_dir / "evaluations.jsonl"
    evaluations_path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n"
                                     for record in all_records))
    summary_path = output_dir / "summary.csv"
    fields = ("operation", "candidate_identity", "implementation_identity", "device_identities",
              "device_sm_versions", "phase", "capacity", "context_length", "rows",
              "dispatch_family", "relative_rms_error", "max_absolute_error", "cosine_similarity",
              "finite_outputs", "contract_status", "evaluation_status", "replayed_metrics_only")
    with summary_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in all_records:
            parameters = record.get("reference_operation_parameters", {})
            metrics = record.get("measured_metrics", record)
            context = record
            writer.writerow({
                "operation": record.get("operation", record.get("contract_proposal_id", "")),
                "candidate_identity": record.get("candidate_identity", ""),
                "implementation_identity": record.get("implementation_identity", ""),
                "device_identities": ";".join(record.get("device_identities", [])),
                "device_sm_versions": ";".join(map(str, record.get("device_sm_versions", []))),
                "phase": context.get("phase", ""), "capacity": context.get("capacity", ""),
                "context_length": context.get("context_length", ""), "rows": context.get("rows", ""),
                "dispatch_family": parameters.get("dispatch_family", ""),
                "relative_rms_error": metrics.get("relative_rms_error", ""),
                "max_absolute_error": metrics.get("max_absolute_error", ""),
                "cosine_similarity": metrics.get("cosine_similarity", ""),
                "finite_outputs": metrics.get("finite_outputs", ""),
                "contract_status": record.get("contract_status", "NEEDS_CALIBRATION"),
                "evaluation_status": record.get("evaluation_status", "NOT_TESTED"),
                "replayed_metrics_only": record.get("replayed_metrics_only", True),
            })

    generated = [p for p in output_dir.rglob("*") if p.is_file() and p.name != "run-manifest.json"]
    provenance["outputs"] = [{"path": str(path.relative_to(output_dir)),
                              "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                             for path in sorted(generated)]
    provenance["evaluation_record_count"] = len(all_records)
    provenance["contracts_observed"] = sorted({record.get("contract_id", record.get("contract_proposal_id", ""))
                                                for record in all_records})
    provenance["result_status"] = "DIAGNOSTIC_ONLY_NEEDS_CALIBRATION"
    provenance["admission_authority_created"] = False
    (output_dir / "run-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"calibration complete: {output_dir}")
    print(f"evaluations={len(all_records)} status=DIAGNOSTIC_ONLY_NEEDS_CALIBRATION")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"qwen3_numerical_calibration=FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
