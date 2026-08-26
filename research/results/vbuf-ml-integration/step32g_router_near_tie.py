"""Bounded router near-tie diagnostics for the Step 32G GLM qualification."""

import hashlib
import importlib.util
import sys

import numpy as np


REFERENCE_PATH = "research/results/vbuf-ml-integration/step32g_bounded_reference.py"
EXPERT_COUNT = 128
TOP_K = 8
SHARED_EXPERT = 2**32 - 1


def load_reference():
    spec = importlib.util.spec_from_file_location("step32g_reference", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Step 32G reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def input_tensor():
    value = np.empty((1, 4, 4096), dtype=np.float32)
    for index in range(value.size):
        x = np.float32(index + 1)
        value.reshape(-1)[index] = np.sin(x * np.float32(0.00017)) * np.float32(0.05) + np.cos(
            x * np.float32(0.000031)
        ) * np.float32(0.01)
    return value


def select(corrected):
    rows = corrected.reshape(-1, corrected.shape[-1])
    return np.asarray(
        [np.argsort(-row, kind="stable")[:TOP_K] for row in rows], dtype=np.int64
    ).reshape(1, 4, TOP_K)


def route_weights(scores, selections):
    weights = np.empty_like(scores[..., :TOP_K])
    for token in range(4):
        denominator = np.sum(scores[0, token, selections[0, token]], dtype=np.float32)
        weights[0, token] = scores[0, token, selections[0, token]] / (denominator + 1e-20)
    return weights


def scalar_linear(value, weight):
    rows = value.reshape(-1, value.shape[-1])
    result = np.empty((rows.shape[0], weight.shape[0]), dtype=np.float32)
    for row in range(rows.shape[0]):
        for output in range(weight.shape[0]):
            total = np.float32(0.0)
            for column in range(rows.shape[1]):
                total = np.float32(
                    total + np.float32(rows[row, column] * weight[output, column])
                )
            result[row, output] = total
    return result.reshape(value.shape[:-1] + (weight.shape[0],))


def router_math(reader, tensors, experts, layer_id, value, scalar=False, high_precision=False):
    router_id = experts[layer_id][(SHARED_EXPERT, 10)][0]
    correction_id = experts[layer_id][(SHARED_EXPERT, 14)][0]
    weight = reader.get(layer_id, router_id)
    correction = reader.get(layer_id, correction_id)
    if high_precision:
        raw = value.astype(np.float64) @ weight.astype(np.float64).T
        scores = 1.0 / (1.0 + np.exp(-raw))
        corrected = scores + correction.astype(np.float64)
        selections = np.asarray(
            [np.argsort(-row, kind="stable")[:TOP_K] for row in corrected.reshape(-1, EXPERT_COUNT)],
            dtype=np.int64,
        ).reshape(1, 4, TOP_K)
        return raw, scores, corrected, selections
    raw = scalar_linear(value, weight) if scalar else reader_module.linear(value, weight)
    scores = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    corrected = scores + correction
    return raw, scores, corrected, select(corrected)


def finite_stats(actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denominator = np.maximum(np.abs(actual.astype(np.float64)), 1e-20)
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "actual_hash": tensor_hash(actual),
        "reference_hash": tensor_hash(reference),
        "actual_l2": float(np.linalg.norm(actual.astype(np.float64))),
        "reference_l2": float(np.linalg.norm(reference.astype(np.float64))),
        "actual_max_abs": float(np.max(np.abs(actual))),
        "reference_max_abs": float(np.max(np.abs(reference))),
        "actual_min": float(np.min(actual)),
        "actual_max": float(np.max(actual)),
        "reference_min": float(np.min(reference)),
        "reference_max": float(np.max(reference)),
        "max_abs_delta": float(np.max(delta)),
        "max_rel_delta": float(np.max(delta / denominator)),
        "mean_abs_delta": float(np.mean(delta)),
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "exact_equal_count": int(np.count_nonzero(actual == reference)),
        "finite_count": int(np.count_nonzero(np.isfinite(actual) & np.isfinite(reference))),
        "nan_count_actual": int(np.count_nonzero(np.isnan(actual))),
        "nan_count_reference": int(np.count_nonzero(np.isnan(reference))),
        "inf_count_actual": int(np.count_nonzero(np.isinf(actual))),
        "inf_count_reference": int(np.count_nonzero(np.isinf(reference))),
    }


def ordered_bits(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return np.where(bits & 0x80000000, ~bits, bits | 0x80000000).astype(np.int64)


def ulp_distance(left, right):
    return int(abs(int(ordered_bits(np.float32(left))) - int(ordered_bits(np.float32(right)))))


def ranked_window(scores, token):
    row = scores.reshape(-1, scores.shape[-1])[token]
    order = np.argsort(-row, kind="stable")
    return {
        "order": order[:11].tolist(),
        "scores": [float(row[index]) for index in order[:11]],
        "rank_k_minus_2": float(row[order[6]]),
        "rank_k_minus_1": float(row[order[7]]),
        "rank_k": float(row[order[7]]),
        "rank_k_plus_1": float(row[order[8]]),
        "rank_k_plus_2": float(row[order[9]]),
        "cutoff_margin": float(row[order[7]] - row[order[8]]),
        "cutoff_margin_ulps": ulp_distance(row[order[7]], row[order[8]]),
    }


def raw_payload_hash(reader, tensors, layer_id, tensor_id):
    _, _, offset, length, scale_id = tensors[layer_id][tensor_id]
    weight_bytes = reader.payload[offset : offset + length].tobytes()
    scale_bytes = b""
    if scale_id:
        _, _, scale_offset, scale_length, _ = tensors[layer_id][scale_id]
        scale_bytes = reader.payload[scale_offset : scale_offset + scale_length].tobytes()
    return hashlib.sha256(weight_bytes).hexdigest(), hashlib.sha256(scale_bytes).hexdigest()


def route_ids(value):
    return value.reshape(4, TOP_K).tolist()


def route_output(reader, tensors, experts, layer_id, normalized, selections, scores):
    output = np.zeros_like(normalized)
    unique_experts = sorted(set(selections.reshape(-1).tolist()))
    for expert in unique_experts:
        gate = reader.get(layer_id, experts[layer_id][(expert, 1)][0])
        up = reader.get(layer_id, experts[layer_id][(expert, 2)][0])
        down = reader.get(layer_id, experts[layer_id][(expert, 3)][0])
        gate_value = reader_module.linear(normalized, gate)
        up_value = reader_module.linear(normalized, up)
        expert_value = reader_module.linear(gate_value / (1.0 + np.exp(-gate_value)) * up_value, down)
        for token in range(4):
            for rank in range(TOP_K):
                if selections[0, token, rank] == expert:
                    denominator = np.sum(scores[0, token, selections[0, token]], dtype=np.float32)
                    output[0, token] += (
                        scores[0, token, expert] / (denominator + 1e-20) * expert_value[0, token]
                    )
    shared_gate = reader_module.linear(
        normalized, reader.get(layer_id, experts[layer_id][(SHARED_EXPERT, 11)][0])
    )
    shared_up = reader_module.linear(
        normalized, reader.get(layer_id, experts[layer_id][(SHARED_EXPERT, 12)][0])
    )
    shared_down = reader.get(layer_id, experts[layer_id][(SHARED_EXPERT, 13)][0])
    shared = reader_module.linear(shared_gate / (1.0 + np.exp(-shared_gate)) * shared_up, shared_down)
    return output + shared


def perturbation(reader, tensors, experts, layer_id, production_input, reference_input, production_ids):
    direction = reference_input - production_input
    direction_l2 = float(np.linalg.norm(direction.astype(np.float64)))
    if direction_l2 == 0.0:
        return {"magnitude": 0.0, "flip_observed": False, "min_flip_l2": None}
    for sign in (1.0, -1.0):
        for factor in [1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0]:
            candidate = production_input + np.float32(sign * factor) * direction
            _, _, _, selected = router_math(reader, tensors, experts, layer_id, candidate, scalar=True)
            if not np.array_equal(selected, production_ids):
                return {
                    "magnitude": float(np.max(np.abs(candidate - production_input))),
                    "flip_observed": True,
                    "min_flip_l2": factor * direction_l2,
                    "factor": factor,
                    "sign": sign,
                }
    return {
        "magnitude": float(np.max(np.abs(direction))),
        "flip_observed": False,
        "min_flip_l2": None,
    }


def main():
    global reader_module
    reader_module = load_reference()
    manifest, payload, checkpoints = sys.argv[1:4]
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 46
    tensors, experts, outputs = reader_module.read_manifest(manifest)
    actual = reader_module.read_records(checkpoints)
    reader = reader_module.Reader(payload, tensors, outputs)
    value = input_tensor()
    natural = {}
    mismatches = []
    for layer_id in range(depth):
        value, selections, stages = reader_module.layer(reader, tensors, experts, layer_id, value)
        natural[layer_id] = (value, selections, stages)
        if selections is not None:
            production = actual[f"layer{layer_id}.selection_ids"].astype(np.int64).reshape(4, TOP_K)
            reference = selections.reshape(4, TOP_K)
            for token in range(4):
                for rank in range(TOP_K):
                    if production[token, rank] != reference[token, rank]:
                        mismatches.append((layer_id, token, rank, production[token], reference[token]))
    print(f"KNOWN_ROUTING_DIVERGENCE_COUNT={len(mismatches)}")
    print(f"TOTAL_ROUTER_INVOCATIONS={max(0, depth - 1) * 4}")
    print(f"TOTAL_ROUTING_SLOTS={max(0, depth - 1) * 4 * TOP_K}")
    print(f"ROUTING_MISMATCH_COUNT={len(mismatches)}")
    for index, (layer_id, token, rank, production_ids, reference_ids) in enumerate(mismatches, 1):
        _, _, stages = natural[layer_id]
        production_all = actual[f"layer{layer_id}.selection_ids"].astype(np.int64).reshape(1, 4, TOP_K)
        reference_all = stages["router_corrected"]
        reference_all = select(reference_all)
        production_input = actual[f"layer{layer_id}.router_input"]
        reference_input = stages["router_input"]
        production_raw = actual[f"layer{layer_id}.router_raw"]
        reference_raw = stages["router_raw"]
        production_scores = actual[f"layer{layer_id}.router_scores"]
        reference_scores = stages["router_scores"]
        production_corrected = actual[f"layer{layer_id}.router_corrected"]
        reference_corrected = stages["router_corrected"]
        print(f"DIVERGENCE_{index}_LAYER={layer_id}")
        print(f"DIVERGENCE_{index}_TOKEN={token}")
        print(f"DIVERGENCE_{index}_RANK_ZERO_BASED={rank}")
        print(f"DIVERGENCE_{index}_PRODUCTION_EXPERT_IDS={production_ids.tolist()}")
        print(f"DIVERGENCE_{index}_REFERENCE_EXPERT_IDS={reference_ids.tolist()}")
        for name, left, right in [
            ("ROUTER_INPUT", production_input, reference_input),
            ("RAW_SCORE", production_raw, reference_raw),
            ("SCORES", production_scores, reference_scores),
            ("CORRECTED", production_corrected, reference_corrected),
        ]:
            print(f"DIVERGENCE_{index}_{name}_STATS={finite_stats(left, right)}")
        p_window = ranked_window(production_corrected, token)
        r_window = ranked_window(reference_corrected, token)
        print(f"DIVERGENCE_{index}_PRODUCTION_CUTOFF={p_window}")
        print(f"DIVERGENCE_{index}_REFERENCE_CUTOFF={r_window}")
        production_set = set(production_ids.tolist())
        reference_set = set(reference_ids.tolist())
        production_only = sorted(production_set - reference_set)
        reference_only = sorted(reference_set - production_set)
        print(f"DIVERGENCE_{index}_PRODUCTION_ONLY={production_only}")
        print(f"DIVERGENCE_{index}_REFERENCE_ONLY={reference_only}")
        for expert in sorted(production_set | reference_set):
            print(
                f"DIVERGENCE_{index}_EXPERT_{expert}_SCORES="
                f"production={float(production_corrected[0, token, expert])} "
                f"reference={float(reference_corrected[0, token, expert])}"
            )
        prod_math = router_math(reader, tensors, experts, layer_id, production_input, scalar=True)
        ref_math_on_prod = router_math(reader, tensors, experts, layer_id, production_input)
        prod_math_on_ref = router_math(reader, tensors, experts, layer_id, reference_input, scalar=True)
        ref_math_on_ref = router_math(reader, tensors, experts, layer_id, reference_input)
        print(f"DIVERGENCE_{index}_PROD_MATH_ON_PROD_INPUT={route_ids(prod_math[3])}")
        print(f"DIVERGENCE_{index}_REF_MATH_ON_PROD_INPUT={route_ids(ref_math_on_prod[3])}")
        print(f"DIVERGENCE_{index}_PROD_MATH_ON_REF_INPUT={route_ids(prod_math_on_ref[3])}")
        print(f"DIVERGENCE_{index}_REF_MATH_ON_REF_INPUT={route_ids(ref_math_on_ref[3])}")
        high = router_math(reader, tensors, experts, layer_id, production_input, high_precision=True)
        high_window = ranked_window(high[2].astype(np.float32), token)
        print(f"DIVERGENCE_{index}_HIGH_PRECISION_EXPERT_IDS={route_ids(high[3])}")
        print(f"DIVERGENCE_{index}_HIGH_PRECISION_CUTOFF={high_window}")
        print(
            f"DIVERGENCE_{index}_HIGH_PRECISION_SWAPPED_SCORES="
            + " ".join(
                f"expert={expert} score={high[2][0, token, expert]:.17g}"
                for expert in sorted(production_set | reference_set)
            )
        )
        print(
            f"DIVERGENCE_{index}_PRODUCTION_WEIGHTS="
                f"{route_weights(production_scores, production_all)[0, token].tolist()}"
        )
        print(
            f"DIVERGENCE_{index}_REFERENCE_WEIGHTS="
            f"{route_weights(reference_scores, reference_all)[0, token].tolist()}"
        )
        corrected_delta = np.max(np.abs(production_corrected - reference_corrected))
        print(
            f"DIVERGENCE_{index}_ERROR_TO_MARGIN_RATIO="
            f"{float(corrected_delta / min(abs(p_window['cutoff_margin']), abs(r_window['cutoff_margin'])))}"
        )
        for expert in sorted(production_set | reference_set):
            print(
                f"DIVERGENCE_{index}_EXPERT_{expert}_ULPS="
                f"{ulp_distance(production_corrected[0, token, expert], reference_corrected[0, token, expert])}"
            )
        layer_input = reference_input
        p_moe = route_output(
            reader, tensors, experts, layer_id, layer_input, production_all,
            prod_math_on_ref[1],
        )
        r_moe = route_output(
            reader, tensors, experts, layer_id, layer_input,
            reference_all,
            ref_math_on_ref[1],
        )
        print(f"DIVERGENCE_{index}_COMMON_INPUT_ROUTE_MOE_DELTA={float(np.max(np.abs(p_moe - r_moe)))}")
        print(f"DIVERGENCE_{index}_PERTURBATION={perturbation(reader, tensors, experts, layer_id, production_input, reference_input, production_all)}")
        weight_id = experts[layer_id][(SHARED_EXPERT, 10)][0]
        print(f"DIVERGENCE_{index}_ROUTER_WEIGHT_ID={weight_id}")
        print(f"DIVERGENCE_{index}_ROUTER_WEIGHT_RAW_HASHES={raw_payload_hash(reader, tensors, layer_id, weight_id)}")
        actual_moe = actual[f"layer{layer_id}.moe_output"]
        print(f"DIVERGENCE_{index}_NATURAL_MOE_OUTPUT_DELTA={float(np.max(np.abs(actual_moe - stages['moe_output'])))}")
        print(f"DIVERGENCE_{index}_NATURAL_LAYER_OUTPUT_DELTA={float(np.max(np.abs(actual[f'layer{layer_id}.output'] - natural[layer_id][0])))}")


if __name__ == "__main__":
    main()
