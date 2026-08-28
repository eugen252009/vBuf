"""Independent bounded GLM-4.5-Air reference for Step 32G/32I checkpoints."""

import gc
import hashlib
import json
import os
import struct
import sys
from collections import defaultdict

import numpy as np


def read_records(path):
    data = open(path, "rb").read()
    pos = 0
    result = {}
    while pos < len(data):
        size = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        name = data[pos : pos + size].decode()
        pos += size
        rank = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dimensions = struct.unpack_from("<" + "Q" * rank, data, pos)
        pos += 8 * rank
        count = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        value = np.frombuffer(data, dtype="<f4", count=count, offset=pos).copy()
        pos += count * 4
        result[name] = value.reshape(dimensions)
    return result


def write_generation_state(path, state):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(state, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_generation_state(path):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            state = json.load(stream)
    except FileNotFoundError:
        return None
    required = {
        "next_step",
        "reference_token_ids",
        "next_reference_token",
        "routing_mismatches",
        "order_mismatches",
        "argmax_mismatches",
        "top10_mismatches",
        "max_abs",
        "max_rel",
        "max_logit_abs",
        "max_logit_rel",
        "step_max_abs",
        "step_logits_abs",
    }
    if not required.issubset(state):
        raise AssertionError("reference state is missing required fields")
    return state


def read_manifest(path):
    tensors = defaultdict(dict)
    experts = defaultdict(dict)
    outputs = {}
    with open(path) as stream:
        next(stream)
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "tensor":
                _, layer, ident, rep, dims, offset, length, scale = fields
                tensors[int(layer)][int(ident)] = (
                    rep,
                    tuple(int(item) for item in dims.split(",")),
                    int(offset),
                    int(length),
                    int(scale),
                )
            elif fields[0] == "expert":
                _, layer, expert, role, ident, scale = fields
                experts[int(layer)][(int(expert), int(role))] = (
                    int(ident),
                    int(scale),
                )
            elif fields[0] == "output":
                _, _, ident, rep, dims, offset, length, _, label = fields
                outputs[label] = (
                    int(ident),
                    rep,
                    tuple(int(item) for item in dims.split(",")),
                    int(offset),
                    int(length),
                )
    return tensors, experts, outputs


def decode_e4m3(bits):
    sign = np.where(bits & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    values = np.where(
        exponent == 0,
        mantissa.astype(np.float32) * np.float32(2.0 ** -9),
        (np.float32(1.0) + mantissa.astype(np.float32) / np.float32(8.0))
        * np.power(np.float32(2.0), exponent.astype(np.float32) - np.float32(7.0)),
    )
    values = sign * values
    values[(exponent == 0x0F) & (mantissa == 0x07)] = np.nan
    return values


class Reader:
    def __init__(self, payload, tensors, outputs):
        self.payload = np.memmap(payload, mode="r", dtype=np.uint8)
        self.tensors = tensors
        self.outputs = outputs

    def decode(self, rep, dims, offset, length, scale_id=None, layer=None):
        raw = self.payload[offset : offset + length]
        if rep == "Bf16":
            bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
            return bits.view("<f4").reshape(dims)
        if rep == "CanonicalPrimitive":
            return np.frombuffer(raw, dtype="<f4").reshape(dims)
        if rep == "F8_E4M3":
            values = decode_e4m3(np.frombuffer(raw, dtype=np.uint8)).reshape(dims)
            scale_rep, scale_dims, scale_offset, scale_length, _ = self.tensors[layer][scale_id]
            if scale_rep != "CanonicalPrimitive":
                raise ValueError("invalid FP8 scale representation")
            scale_raw = self.payload[scale_offset : scale_offset + scale_length]
            scales = np.frombuffer(scale_raw, dtype="<f4").reshape((dims[0], 1))
            return values * scales
        raise ValueError(rep)

    def get(self, layer, ident):
        rep, dims, offset, length, scale_id = self.tensors[layer][ident]
        return self.decode(rep, dims, offset, length, scale_id, layer)

    def output(self, label):
        _, rep, dims, offset, length = self.outputs[label]
        return self.decode(rep, dims, offset, length)

    def bf16_rows(self, label, token_ids):
        _, rep, dims, offset, length = self.outputs[label]
        if rep != "Bf16" or len(dims) != 2 or dims[1] != 4096:
            raise ValueError("embedding tensor is not the qualified BF16 matrix")
        row_bytes = dims[1] * 2
        if length != dims[0] * row_bytes:
            raise ValueError("embedding tensor payload length is invalid")
        rows = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= dims[0]:
                raise ValueError("embedding token ID is outside vocabulary")
            raw = self.payload[offset + token_id * row_bytes : offset + (token_id + 1) * row_bytes]
            bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
            rows.append(bits.view("<f4"))
        return np.asarray(rows, dtype=np.float32).reshape(1, len(token_ids), dims[1])


def rms(value, weight):
    width = value.shape[-1]
    mean = np.sum(value * value, axis=-1, keepdims=True, dtype=np.float32) / np.float32(width)
    return value * (mean + np.float32(1e-5)) ** np.float32(-0.5) * weight


def linear(value, weight):
    shape = value.shape[:-1] + (weight.shape[0],)
    return (value.reshape(-1, value.shape[-1]) @ weight.T).reshape(shape)


def apply_rope(value, heads):
    result = value.copy()
    for index in range(32):
        inverse = np.float32(1_000_000.0) ** np.float32(-2.0 * index / 64.0)
        for position in range(value.shape[1]):
            angle = np.float32(position) * inverse
            cosine, sine = np.cos(angle), np.sin(angle)
            for head in range(heads):
                base = (position * heads + head) * 128
                left = value[0, position, head, index]
                right = value[0, position, head, 32 + index]
                result[0, position, head, index] = left * cosine - right * sine
                result[0, position, head, 32 + index] = right * cosine + left * sine
    return result


def attention(q, k, v, rotated=False):
    if not rotated:
        q, k = apply_rope(q, 96), apply_rope(k, 8)
    result = np.empty_like(q)
    sequence = q.shape[1]
    for position in range(sequence):
        for head in range(96):
            kv_head = head // 12
            scores = np.empty(position + 1, dtype=np.float32)
            for key_position in range(position + 1):
                scores[key_position] = np.dot(
                    q[0, position, head], k[0, key_position, kv_head]
                ) * np.float32(1.0 / np.sqrt(128.0))
            scores -= np.max(scores)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, dtype=np.float32)
            for component in range(128):
                result[0, position, head, component] = np.sum(
                    probabilities * v[0, : position + 1, kv_head, component],
                    dtype=np.float32,
                )
    return result


def layer(reader, tensors, experts, layer_id, value):
    sequence = value.shape[1]
    if layer_id == 0:
        base = 2
        normalized = rms(value, reader.get(0, base))
        q = linear(normalized, reader.get(0, base + 14)) + reader.get(0, base + 13)
        k = linear(normalized, reader.get(0, base + 9)) + reader.get(0, base + 8)
        v = linear(normalized, reader.get(0, base + 17)) + reader.get(0, base + 16)
        q_heads = apply_rope(q.reshape(1, sequence, 96, 128), 96)
        k_heads = apply_rope(k.reshape(1, sequence, 8, 128), 8)
        v_heads = v.reshape(1, sequence, 8, 128)
        attended = attention(q_heads, k_heads, v_heads, rotated=True).reshape(1, sequence, 12288)
        residual = value + linear(attended, reader.get(0, base + 11))
        normalized = rms(residual, reader.get(0, base + 7))
        gate = linear(normalized, reader.get(0, base + 3))
        up = linear(normalized, reader.get(0, base + 5))
        output = residual + linear(gate / (1 + np.exp(-gate)) * up, reader.get(0, base + 1))
        return output, None, {
            "kv_key": k_heads,
            "kv_value": v_heads,
            "post_attention_residual": residual,
        }

    ids = [ident for pair in experts[layer_id].values() for ident in pair if ident != 0]
    post_norm = max(ids) + 1
    normalized = rms(value, reader.get(layer_id, min(ids) - 1))
    q = linear(normalized, reader.get(layer_id, post_norm + 7)) + reader.get(layer_id, post_norm + 6)
    k = linear(normalized, reader.get(layer_id, post_norm + 2)) + reader.get(layer_id, post_norm + 1)
    v = linear(normalized, reader.get(layer_id, post_norm + 10)) + reader.get(layer_id, post_norm + 9)
    q_heads = apply_rope(q.reshape(1, sequence, 96, 128), 96)
    k_heads = apply_rope(k.reshape(1, sequence, 8, 128), 8)
    v_heads = v.reshape(1, sequence, 8, 128)
    attended = attention(q_heads, k_heads, v_heads, rotated=True).reshape(1, sequence, 12288)
    residual = value + linear(attended, reader.get(layer_id, post_norm + 4))
    normalized = rms(residual, reader.get(layer_id, post_norm))
    shared_key = 2**32 - 1
    raw = linear(normalized, reader.get(layer_id, experts[layer_id][(shared_key, 10)][0]))
    scores = 1.0 / (1.0 + np.exp(-raw))
    corrected = scores + reader.get(layer_id, experts[layer_id][(shared_key, 14)][0])
    selections = []
    for row in corrected.reshape(-1, corrected.shape[-1]):
        selections.extend(np.argsort(-row, kind="stable")[:8].tolist())
    selections = np.asarray(selections, dtype=np.int64).reshape(sequence, 8)
    routed = np.zeros_like(normalized)
    expert_stages = {}
    for expert in sorted(set(selections.reshape(-1).tolist())):
        gate_id = experts[layer_id][(expert, 1)][0]
        up_id = experts[layer_id][(expert, 2)][0]
        down_id = experts[layer_id][(expert, 3)][0]
        gate = linear(normalized, reader.get(layer_id, gate_id))
        up = linear(normalized, reader.get(layer_id, up_id))
        expert_value = linear(gate / (1 + np.exp(-gate)) * up, reader.get(layer_id, down_id))
        dispatched = np.zeros_like(expert_value)
        for token in range(sequence):
            for rank in range(8):
                if selections[token, rank] == expert:
                    denominator = np.sum(scores[0, token, selections[token]], dtype=np.float32)
                    dispatched[0, token] += scores[0, token, expert] / (denominator + 1e-20) * expert_value[0, token]
        routed += dispatched
        expert_stages[f"expert_{expert}"] = dispatched
    shared_gate = linear(normalized, reader.get(layer_id, experts[layer_id][(shared_key, 11)][0]))
    shared_up = linear(normalized, reader.get(layer_id, experts[layer_id][(shared_key, 12)][0]))
    shared_down = reader.get(layer_id, experts[layer_id][(shared_key, 13)][0])
    shared_multiply = shared_gate / (1 + np.exp(-shared_gate)) * shared_up
    shared = linear(shared_multiply, shared_down)
    return residual + routed + shared, selections, {
        "kv_key": k_heads,
        "kv_value": v_heads,
        "post_attention_residual": residual,
        "router_input": normalized,
        "router_raw": raw,
        "router_scores": scores,
        "router_corrected": corrected,
        "shared_expert_output": shared,
        "shared_gate": shared_gate,
        "shared_up": shared_up,
        "shared_multiply": shared_multiply,
        "moe_output": routed + shared,
        **expert_stages,
    }


def generation_reference(actual, reader, tensors, experts, depth, max_steps, state_path=None):
    prefill_token_ids = actual["prefill.input_token_ids"].astype(np.int64).reshape(-1)
    if not np.array_equal(prefill_token_ids, [51, 68, 82, 83]):
        raise AssertionError(f"token ID reference mismatch: {prefill_token_ids.tolist()}")
    production_token_ids = actual["generation.generated_token_ids"].astype(np.int64).reshape(-1)
    if production_token_ids.size != max_steps:
        raise AssertionError(
            f"production generated {production_token_ids.size} tokens, expected {max_steps}"
        )
    state = read_generation_state(state_path)
    if state is None:
        start_step = 0
        reference_token_ids = prefill_token_ids.tolist()
        max_abs = max_rel = 0.0
        max_logit_abs = max_logit_rel = 0.0
        routing_mismatches = 0
        order_mismatches = 0
        argmax_mismatches = 0
        top10_mismatches = 0
        step_max_abs = []
        step_max_logit_abs = []
    else:
        start_step = int(state["next_step"])
        reference_token_ids = [int(token) for token in state["reference_token_ids"]]
        if len(reference_token_ids) != len(prefill_token_ids) + start_step:
            raise AssertionError("reference state token count is inconsistent")
        max_abs = float(state["max_abs"])
        max_rel = float(state["max_rel"])
        max_logit_abs = float(state["max_logit_abs"])
        max_logit_rel = float(state["max_logit_rel"])
        routing_mismatches = int(state["routing_mismatches"])
        order_mismatches = int(state["order_mismatches"])
        argmax_mismatches = int(state["argmax_mismatches"])
        top10_mismatches = int(state["top10_mismatches"])
        step_max_abs = [float(value) for value in state["step_max_abs"]]
        step_max_logit_abs = [float(value) for value in state["step_logits_abs"]]
    prefill_value = reader.bf16_rows("embedding", np.asarray(reference_token_ids, dtype=np.int64))
    for layer_id in range(depth):
        prefill_value, _, _ = layer(reader, tensors, experts, layer_id, prefill_value)
    prefill_logits = linear(rms(prefill_value, reader.output("final_norm")), reader.output("lm_head"))
    next_reference_token = (
        int(state["next_reference_token"])
        if state is not None
        else int(np.argmax(prefill_logits[0, -1]))
    )
    for step in range(start_step, max_steps):
        token_ids = np.asarray(reference_token_ids, dtype=np.int64)
        consumed_token = next_reference_token
        token_ids = np.concatenate((token_ids, [consumed_token]))
        value = reader.bf16_rows("embedding", token_ids)
        step_max = 0.0
        step_rel = 0.0
        for layer_id in range(depth):
            value, selections, stages = layer(reader, tensors, experts, layer_id, value)
            prefix = f"generation.step{step}.layer{layer_id}"
            output = actual[f"{prefix}.output"]
            delta = np.abs(value[:, -1:] - output)
            step_max = max(step_max, float(np.max(delta)))
            step_rel = max(
                step_rel,
                float(np.max(delta / np.maximum(np.abs(output), 1e-20))),
            )
            if selections is not None:
                expected = actual[f"{prefix}.selection_ids"].astype(np.int64)
                reference_selection = selections[None, -1:]
                routing_mismatches += int(np.count_nonzero(expected != reference_selection))
                production_order = expected.reshape(-1).tolist()
                reference_order = reference_selection.reshape(-1).tolist()
                order_mismatches += int(production_order != reference_order)
            for name, reference in stages.items():
                expected_stage = actual.get(f"{prefix}.{name}")
                if expected_stage is None:
                    continue
                if name not in {"kv_key", "kv_value"}:
                    reference = reference[:, -1:]
                stage_delta = np.abs(reference - expected_stage)
                step_max = max(step_max, float(np.max(stage_delta)))
            print(
                f"generation_step={step + 1} layer={layer_id} "
                f"max_abs={step_max:.8e} max_rel={step_rel:.8e}",
                flush=True,
            )
        actual_logits = actual[f"generation.step{step}.logits"]
        final_norm = rms(value, reader.output("final_norm"))
        logits = linear(final_norm, reader.output("lm_head"))
        expected_logits = actual_logits
        logit_delta = np.abs(logits[:, -1:] - expected_logits)
        logit_abs = float(np.max(logit_delta))
        logit_rel = float(np.max(logit_delta / np.maximum(np.abs(expected_logits), 1e-20)))
        max_logit_abs = max(max_logit_abs, logit_abs)
        max_logit_rel = max(max_logit_rel, logit_rel)
        max_abs = max(max_abs, step_max)
        max_rel = max(max_rel, step_rel)
        step_max_abs.append(step_max)
        step_max_logit_abs.append(logit_abs)
        production_token = int(production_token_ids[step])
        reference_token = int(np.argmax(logits[0, -1]))
        if production_token != consumed_token:
            raise AssertionError(
                f"production/reference consumed token mismatch at step {step + 1}: "
                f"{production_token} != {consumed_token}"
            )
        production_next_token = int(np.argmax(expected_logits[0, -1]))
        if production_next_token != reference_token:
            argmax_mismatches += 1
        production_top10 = np.argsort(-expected_logits[0, -1], kind="stable")[:10]
        reference_top10 = np.argsort(-logits[0, -1], kind="stable")[:10]
        if not np.array_equal(production_top10, reference_top10):
            top10_mismatches += 1
        print(
            f"generation_step_summary={step + 1} production_token={production_token} "
            f"reference_consumed_token={consumed_token} production_next_token={production_next_token} "
            f"reference_next_token={reference_token} logits_max_abs={logit_abs:.8e} "
            f"logits_max_rel={logit_rel:.8e} production_top10={production_top10.tolist()} "
            f"reference_top10={reference_top10.tolist()}",
            flush=True,
        )
        reference_token_ids.append(consumed_token)
        next_reference_token = reference_token
        if state_path:
            write_generation_state(
                state_path,
                {
                    "next_step": step + 1,
                    "reference_token_ids": reference_token_ids,
                    "next_reference_token": next_reference_token,
                    "routing_mismatches": routing_mismatches,
                    "order_mismatches": order_mismatches,
                    "argmax_mismatches": argmax_mismatches,
                    "top10_mismatches": top10_mismatches,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "max_logit_abs": max_logit_abs,
                    "max_logit_rel": max_logit_rel,
                    "step_max_abs": step_max_abs,
                    "step_logits_abs": step_max_logit_abs,
                },
            )
    generated_reference = reference_token_ids[len(prefill_token_ids) :]
    print(f"reference_generated_token_ids={generated_reference}")
    print(f"production_generated_token_ids={production_token_ids.tolist()}")
    print(f"generated_token_sequence_parity={generated_reference == production_token_ids.tolist()}")
    print(f"routing_membership_mismatches={routing_mismatches}")
    print(f"routing_order_mismatches={order_mismatches}")
    print(f"argmax_mismatches={argmax_mismatches}")
    print(f"top10_mismatches={top10_mismatches}")
    print(f"max_transformer_abs={max_abs:.8e} max_transformer_rel={max_rel:.8e}")
    print(f"max_step_logits_abs={max_logit_abs:.8e} max_step_logits_rel={max_logit_rel:.8e}")
    print(f"step_transformer_abs={step_max_abs}")
    print(f"step_logits_abs={step_max_logit_abs}")


def main():
    manifest_path, payload_path, checkpoint_path = sys.argv[1:4]
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 46
    input_mode = sys.argv[5] if len(sys.argv) > 5 else "synthetic"
    real_input = input_mode == "real"
    decode_input = input_mode == "decode"
    qualification_text = sys.argv[6] if len(sys.argv) > 6 else "Test"
    state_path = None
    if len(sys.argv) > 8:
        if sys.argv[8] != "--state" or len(sys.argv) != 10:
            raise AssertionError("invalid reference state arguments")
        state_path = sys.argv[9]
    tensors, experts, outputs = read_manifest(manifest_path)
    actual = read_records(checkpoint_path)
    reader = Reader(payload_path, tensors, outputs)
    if input_mode == "generation":
        generation_reference(
            actual, reader, tensors, experts, depth, int(sys.argv[7]), state_path
        )
        return
    if real_input or decode_input:
        prefill_token_ids = actual["prefill.input_token_ids" if decode_input else "input_token_ids"].astype(np.int64).reshape(-1)
        expected_token_ids = np.asarray([51, 68, 82, 83], dtype=np.int64)
        token_ids_pass = np.array_equal(prefill_token_ids, expected_token_ids)
        if not token_ids_pass:
            raise AssertionError(f"token ID reference mismatch: {prefill_token_ids.tolist()}")
        token_ids = np.concatenate((prefill_token_ids, [220])) if decode_input else prefill_token_ids
        value = reader.bf16_rows("embedding", token_ids)
        embedding_key = "input_embedding" if real_input else "prefill.input_embedding"
        embedding_delta = np.abs(value[:, :4] - actual[embedding_key]) if decode_input else np.abs(value - actual[embedding_key])
        decode_embedding_delta = (
            np.abs(value[:, 4:] - actual["decode.input_embedding"])
            if decode_input
            else np.empty(0, dtype=np.float32)
        )
        print(f"real_text={qualification_text!r}")
        print(f"raw_text_utf8_bytes={len(qualification_text.encode('utf-8'))}")
        print(f"raw_text_sha256={hashlib.sha256(qualification_text.encode('utf-8')).hexdigest()}")
        print(f"token_ids={token_ids.tolist()}")
        print(f"token_ids_hash={hashlib.sha256(token_ids.astype('<f4').tobytes()).hexdigest()}")
        print("special_tokens_added=[]")
        print("persisted_tokenizer_reference=captured-independent-fixture")
        print(f"token_ids_reference_pass={token_ids_pass}")
        print(f"embedding_max_abs={np.max(embedding_delta):.8e}")
        print(f"embedding_max_rel={np.max(embedding_delta / np.maximum(np.abs(actual[embedding_key]), 1e-20)):.8e}")
        print(f"embedding_reference_pass={np.max(embedding_delta) == 0.0}")
        if decode_input:
            print(f"decode_embedding_max_abs={np.max(decode_embedding_delta):.8e}")
            print(f"decode_embedding_reference_pass={np.max(decode_embedding_delta) == 0.0}")
    else:
        value = np.empty((1, 4, 4096), dtype=np.float32)
        for index in range(value.size):
            x = np.float32(index + 1)
            value.reshape(-1)[index] = np.sin(x * np.float32(0.00017)) * np.float32(0.05) + np.cos(x * np.float32(0.000031)) * np.float32(0.01)
    max_abs = max_rel = 0.0
    routing_mismatches = 0
    actual_prefix = "decode." if decode_input else ""
    for layer_id in range(depth):
        value, selections, stages = layer(reader, tensors, experts, layer_id, value)
        output = actual[f"{actual_prefix}layer{layer_id}.output"]
        expected_value = value[:, -1:] if decode_input else value
        delta = np.abs(value - output)
        if decode_input:
            delta = np.abs(expected_value - output)
        max_abs = max(max_abs, float(np.max(delta)))
        max_rel = max(max_rel, float(np.max(delta / np.maximum(np.abs(output), 1e-20))))
        if selections is not None:
            expected = actual[f"{actual_prefix}layer{layer_id}.selection_ids"].astype(np.int64)
            reference_selection = selections[None, -1:] if decode_input else selections
            routing_mismatches += int(np.count_nonzero(expected != reference_selection.reshape(expected.shape)))
        for name, reference in stages.items():
            expected_stage = actual.get(f"{actual_prefix}layer{layer_id}.{name}")
            if expected_stage is not None:
                if decode_input and name not in {"kv_key", "kv_value"}:
                    reference = reference[:, -1:]
                stage_delta = np.abs(reference - expected_stage)
                print(
                    f"layer={layer_id} stage={name} max_abs={np.max(stage_delta):.8e}",
                    flush=True,
                )
        print(f"layer={layer_id} max_abs={np.max(delta):.8e} max_rel={np.max(delta / np.maximum(np.abs(output), 1e-20)):.8e}", flush=True)
        gc.collect()
    final_norm = rms(value, reader.output("final_norm"))
    logits = linear(final_norm, reader.output("lm_head"))
    actual_norm = actual["decode.final_norm"] if decode_input else actual["final_norm"]
    actual_logits = actual["decode.logits"] if decode_input else actual["logits"]
    if decode_input:
        final_norm = final_norm[:, -1:]
        logits = logits[:, -1:]
    norm_delta = np.abs(final_norm - actual_norm)
    logit_delta = np.abs(logits - actual_logits)
    print(f"routing_mismatches={routing_mismatches}")
    print(f"transformer_max_abs={max_abs:.8e} transformer_max_rel={max_rel:.8e}")
    print(f"final_norm_max_abs={np.max(norm_delta):.8e} final_norm_max_rel={np.max(norm_delta / np.maximum(np.abs(actual_norm), 1e-20)):.8e}")
    print(f"logits_max_abs={np.max(logit_delta):.8e} logits_max_rel={np.max(logit_delta / np.maximum(np.abs(actual_logits), 1e-20)):.8e}")
    generic = np.argsort(-logits[0, -1])[:10]
    reference = np.argsort(-actual_logits[0, -1])[:10]
    print(f"generic_argmax={int(generic[0])} reference_argmax={int(reference[0])}")
    print(f"generic_top5={generic[:5].tolist()} reference_top5={reference[:5].tolist()}")
    print(f"generic_top10={generic.tolist()} reference_top10={reference.tolist()}")


if __name__ == "__main__":
    main()
