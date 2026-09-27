#!/usr/bin/env python3
"""Compare identical-token Qwen3 GGUF/vBuf qualification exports."""

import argparse
import json
from pathlib import Path

import numpy as np


PREFIX_CANDIDATES = (1, 4, 5, 8, 9, 16, 17, 24, 25, 32)
EMBEDDING = 5120
VOCABULARY = 151936


def read_manifest(root: Path, tokens: list[int]) -> tuple[dict[str, str], bool]:
    path = root / "sequence.meta"
    if not path.exists():
        path = root / "reference.meta"
    if not path.exists():
        raise ValueError(f"missing sequence/reference metadata in {root}")
    metadata = dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )
    actual_tokens = [int(value) for value in metadata["tokens"].split(",")]
    if int(metadata["positions"]) != len(tokens) or actual_tokens != tokens:
        raise ValueError(f"token/position mismatch in {root}")
    return metadata, path.name == "reference.meta"


def read_exports(root: Path, tokens: list[int]) -> dict[str, np.ndarray]:
    metadata, llama_format = read_manifest(root, tokens)
    positions = len(tokens)
    embedding = int(metadata.get("embedding", EMBEDDING))
    vocabulary = int(metadata.get("vocabulary", VOCABULARY))
    if embedding != EMBEDDING or vocabulary != VOCABULARY:
        raise ValueError(f"unexpected Qwen3 output dimensions in {root}")
    filenames = (
        ("hidden", "l_out-39-0" if llama_format else "final_hidden", embedding),
        ("norm", "result_norm-0" if llama_format else "final_norm", embedding),
        ("logits", "result_output-0" if llama_format else "logits", vocabulary),
    )
    arrays = {}
    for key, filename, width in filenames:
        path = root / f"{filename}.f32"
        array = np.fromfile(path, dtype="<f4")
        if array.size != positions * width:
            raise ValueError(
                f"bad element count in {path}: {array.size}, expected {positions * width}"
            )
        arrays[key] = array.reshape(positions, width)
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"non-finite values in {path}")
    return arrays


def vector_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | bool | None]:
    x = np.asarray(actual, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    difference = x - y
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    rms = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(y * y)))
    return {
        "all_finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rms": rms,
        "relative_rms": rms / reference_rms if reference_rms else None,
        "cosine": float(np.dot(x, y) / (x_norm * y_norm)) if x_norm and y_norm else None,
        "norm_ratio": x_norm / y_norm if y_norm else None,
    }


def topk(values: np.ndarray, count: int = 10) -> list[int]:
    count = min(count, values.size)
    candidates = np.argpartition(-values, count - 1)[:count]
    return sorted((int(i) for i in candidates), key=lambda i: (-float(values[i]), i))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="BF16 llama.cpp export directory")
    parser.add_argument("--llama-q4", required=True, help="llama.cpp Q4_K_M export directory")
    parser.add_argument("--vbuf-q4", required=True, help="vBuf Q4_K_M export directory")
    parser.add_argument("--tokens", required=True, help="comma-separated exact token IDs")
    parser.add_argument("--output", required=True, help="JSON result path")
    args = parser.parse_args()

    tokens = [int(value) for value in args.tokens.split(",")]
    if not tokens:
        raise SystemExit("at least one token ID is required")
    models = {
        "reference": read_exports(Path(args.reference), tokens),
        "llama_q4": read_exports(Path(args.llama_q4), tokens),
        "vbuf_q4": read_exports(Path(args.vbuf_q4), tokens),
    }
    prefixes = [value for value in PREFIX_CANDIDATES if value <= len(tokens)]
    result: dict[str, object] = {"tokens": tokens, "prefixes": {}}

    for prefix in prefixes:
        row_index = prefix - 1
        reference = models["reference"]
        llama = models["llama_q4"]
        vbuf = models["vbuf_q4"]
        reference_logits = reference["logits"][row_index]
        ref_top = topk(reference_logits)
        row: dict[str, object] = {
            "reference_top10": [
                [token, float(reference_logits[token])] for token in ref_top
            ],
            "reference_margin": float(
                reference_logits[ref_top[0]] - reference_logits[ref_top[1]]
            ),
        }
        for state in ("hidden", "norm", "logits"):
            reference_values = reference[state][row_index]
            llama_values = llama[state][row_index]
            vbuf_values = vbuf[state][row_index]
            llama_metrics = vector_metrics(llama_values, reference_values)
            vbuf_metrics = vector_metrics(vbuf_values, reference_values)
            state_result: dict[str, object] = {
                "llama_q4_vs_reference": llama_metrics,
                "vbuf_q4_vs_reference": vbuf_metrics,
                "llama_q4_vs_vbuf_q4": vector_metrics(llama_values, vbuf_values),
                "vbuf_over_llama_rms_ratio": (
                    vbuf_metrics["rms"] / llama_metrics["rms"]
                    if llama_metrics["rms"]
                    else None
                ),
            }
            if state == "logits":
                llama_top = topk(llama_values)
                vbuf_top = topk(vbuf_values)
                state_result["topk"] = {
                    "reference": ref_top,
                    "llama_q4": llama_top,
                    "vbuf_q4": vbuf_top,
                    "llama_top1_match": llama_top[0] == ref_top[0],
                    "vbuf_top1_match": vbuf_top[0] == ref_top[0],
                    "llama_top5_overlap": len(set(llama_top[:5]) & set(ref_top[:5])),
                    "vbuf_top5_overlap": len(set(vbuf_top[:5]) & set(ref_top[:5])),
                    "llama_top10_overlap": len(set(llama_top) & set(ref_top)),
                    "vbuf_top10_overlap": len(set(vbuf_top) & set(ref_top)),
                    "llama_top10_logits": [
                        [token, float(llama_values[token])] for token in llama_top
                    ],
                    "vbuf_top10_logits": [
                        [token, float(vbuf_values[token])] for token in vbuf_top
                    ],
                    "reference_top1_logit": float(reference_logits[ref_top[0]]),
                    "reference_top2_logit": float(reference_logits[ref_top[1]]),
                    "reference_top1_top2_margin": float(
                        reference_logits[ref_top[0]] - reference_logits[ref_top[1]]
                    ),
                    "llama_argmax": llama_top[0],
                    "vbuf_argmax": vbuf_top[0],
                }
            row[state] = state_result
        result["prefixes"][str(prefix)] = row

    # At input position i, logits score the next fixed token tokens[i + 1].
    if len(tokens) > 1:
        result["teacher_forced_nll"] = {}
        targets = np.asarray(tokens[1:], dtype=np.int64)
        for name, data in models.items():
            rows = data["logits"][:-1].astype(np.float64)
            maxima = rows.max(axis=1)
            nll = maxima + np.log(np.exp(rows - maxima[:, None]).sum(axis=1))
            nll -= rows[np.arange(len(targets)), targets]
            result["teacher_forced_nll"][name] = {
                "tokens_scored": len(targets),
                "mean_nll": float(nll.mean()),
                "perplexity": float(np.exp(nll.mean())),
                "token_nll": nll.tolist(),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    for prefix in prefixes:
        row = result["prefixes"][str(prefix)]
        hidden = row["hidden"]
        logits = row["logits"]
        top = logits["topk"]
        print(
            f"prefix={prefix} hidden_rms llama={hidden['llama_q4_vs_reference']['rms']:.7g} "
            f"vbuf={hidden['vbuf_q4_vs_reference']['rms']:.7g} "
            f"ratio={hidden['vbuf_over_llama_rms_ratio']:.5g}; "
            f"logit_rms llama={logits['llama_q4_vs_reference']['rms']:.7g} "
            f"vbuf={logits['vbuf_q4_vs_reference']['rms']:.7g} "
            f"ratio={logits['vbuf_over_llama_rms_ratio']:.5g}; "
            f"top1 ref/llama/vbuf={top['reference'][0]}/{top['llama_q4'][0]}/"
            f"{top['vbuf_q4'][0]} margin={top['reference_top1_top2_margin']:.7g}"
        )
    print(
        "teacher_forced_nll",
        {
            name: {key: data[key] for key in ("mean_nll", "perplexity")}
            for name, data in result.get("teacher_forced_nll", {}).items()
        },
    )


if __name__ == "__main__":
    main()
