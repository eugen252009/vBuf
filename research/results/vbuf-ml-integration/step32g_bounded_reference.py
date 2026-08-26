"""Independent bounded GLM-4.5-Air reference for Step 32G checkpoints."""

import gc
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


def attention(q, k, v):
    q, k = apply_rope(q, 96), apply_rope(k, 8)
    result = np.empty_like(q)
    for position in range(4):
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
    if layer_id == 0:
        base = 2
        normalized = rms(value, reader.get(0, base))
        q = linear(normalized, reader.get(0, base + 14)) + reader.get(0, base + 13)
        k = linear(normalized, reader.get(0, base + 9)) + reader.get(0, base + 8)
        v = linear(normalized, reader.get(0, base + 17)) + reader.get(0, base + 16)
        attended = attention(
            q.reshape(1, 4, 96, 128),
            k.reshape(1, 4, 8, 128),
            v.reshape(1, 4, 8, 128),
        ).reshape(1, 4, 12288)
        residual = value + linear(attended, reader.get(0, base + 11))
        normalized = rms(residual, reader.get(0, base + 7))
        gate = linear(normalized, reader.get(0, base + 3))
        up = linear(normalized, reader.get(0, base + 5))
        output = residual + linear(gate / (1 + np.exp(-gate)) * up, reader.get(0, base + 1))
        return output, None, {"post_attention_residual": residual}

    ids = [ident for pair in experts[layer_id].values() for ident in pair if ident != 0]
    post_norm = max(ids) + 1
    normalized = rms(value, reader.get(layer_id, min(ids) - 1))
    q = linear(normalized, reader.get(layer_id, post_norm + 7)) + reader.get(layer_id, post_norm + 6)
    k = linear(normalized, reader.get(layer_id, post_norm + 2)) + reader.get(layer_id, post_norm + 1)
    v = linear(normalized, reader.get(layer_id, post_norm + 10)) + reader.get(layer_id, post_norm + 9)
    attended = attention(
        q.reshape(1, 4, 96, 128),
        k.reshape(1, 4, 8, 128),
        v.reshape(1, 4, 8, 128),
    ).reshape(1, 4, 12288)
    residual = value + linear(attended, reader.get(layer_id, post_norm + 4))
    normalized = rms(residual, reader.get(layer_id, post_norm))
    shared_key = 2**32 - 1
    raw = linear(normalized, reader.get(layer_id, experts[layer_id][(shared_key, 10)][0]))
    scores = 1.0 / (1.0 + np.exp(-raw))
    corrected = scores + reader.get(layer_id, experts[layer_id][(shared_key, 14)][0])
    selections = []
    for row in corrected.reshape(-1, corrected.shape[-1]):
        selections.extend(np.argsort(-row, kind="stable")[:8].tolist())
    selections = np.asarray(selections, dtype=np.int64).reshape(4, 8)
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
        for token in range(4):
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
        "post_attention_residual": residual,
        "router_corrected": corrected,
        "shared_expert_output": shared,
        "shared_gate": shared_gate,
        "shared_up": shared_up,
        "shared_multiply": shared_multiply,
        "moe_output": routed + shared,
        **expert_stages,
    }


def main():
    manifest_path, payload_path, checkpoint_path = sys.argv[1:4]
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 46
    tensors, experts, outputs = read_manifest(manifest_path)
    actual = read_records(checkpoint_path)
    reader = Reader(payload_path, tensors, outputs)
    value = np.empty((1, 4, 4096), dtype=np.float32)
    for index in range(value.size):
        x = np.float32(index + 1)
        value.reshape(-1)[index] = np.sin(x * np.float32(0.00017)) * np.float32(0.05) + np.cos(x * np.float32(0.000031)) * np.float32(0.01)
    max_abs = max_rel = 0.0
    routing_mismatches = 0
    for layer_id in range(depth):
        value, selections, stages = layer(reader, tensors, experts, layer_id, value)
        output = actual[f"layer{layer_id}.output"]
        delta = np.abs(value - output)
        max_abs = max(max_abs, float(np.max(delta)))
        max_rel = max(max_rel, float(np.max(delta / np.maximum(np.abs(output), 1e-20))))
        if selections is not None:
            expected = actual[f"layer{layer_id}.selection_ids"].astype(np.int64)
            routing_mismatches += int(np.count_nonzero(expected != selections.reshape(1, 4, 8)))
        for name, reference in stages.items():
            expected_stage = actual.get(f"layer{layer_id}.{name}")
            if expected_stage is not None:
                stage_delta = np.abs(reference - expected_stage)
                print(
                    f"layer={layer_id} stage={name} max_abs={np.max(stage_delta):.8e}",
                    flush=True,
                )
        print(f"layer={layer_id} max_abs={np.max(delta):.8e} max_rel={np.max(delta / np.maximum(np.abs(output), 1e-20)):.8e}", flush=True)
        gc.collect()
    final_norm = rms(value, reader.output("final_norm"))
    logits = linear(final_norm, reader.output("lm_head"))
    norm_delta = np.abs(final_norm - actual["final_norm"])
    logit_delta = np.abs(logits - actual["logits"])
    print(f"routing_mismatches={routing_mismatches}")
    print(f"transformer_max_abs={max_abs:.8e} transformer_max_rel={max_rel:.8e}")
    print(f"final_norm_max_abs={np.max(norm_delta):.8e} final_norm_max_rel={np.max(norm_delta / np.maximum(np.abs(actual['final_norm']), 1e-20)):.8e}")
    print(f"logits_max_abs={np.max(logit_delta):.8e} logits_max_rel={np.max(logit_delta / np.maximum(np.abs(actual['logits']), 1e-20)):.8e}")
    generic = np.argsort(-logits[0, -1])[:10]
    reference = np.argsort(-actual["logits"][0, -1])[:10]
    print(f"generic_argmax={int(generic[0])} reference_argmax={int(reference[0])}")
    print(f"generic_top5={generic[:5].tolist()} reference_top5={reference[:5].tolist()}")
    print(f"generic_top10={generic.tolist()} reference_top10={reference.tolist()}")


if __name__ == "__main__":
    main()
