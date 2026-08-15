#!/usr/bin/env python3
"""CCC structured-baseline qualification (Q8-relative, single tensor).

Research-only, isolated. Reads the pinned Qwen3-32B-Q8_0 GGUF, extracts
blk.0.attn_k.weight, evaluates structured baseline predictors B0-B5 with
small C3 residual alphabets, per-block scale variants, and canonical GGML
controls produced by the pinned llama.cpp (commit 4c1a0af40 + vBuf patches).

Ground truth: reconstructed Q8_0 weights. NOT BF16/F32 truth.
No vBuf/vBuf-ML wire modification; no canonical artifact mutation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, struct, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualify_step16 import parse

ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c"
GGUF_SHA256 = "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"
TENSOR_NAME = "blk.0.attn_k.weight"

Q8_0 = (32, 34, 8, 8)     # weights/block, bytes/block, ggml type id, code bits
Q4_0 = (32, 18, 2, 4)
Q2_K = (256, 84, 10, 2)
Q3_K = (256, 110, 11, 3)

GAMMA_GRID = [1.0, 1.15, 1.25, 1.35, 1.5, 1.7, 2.0]
TAIL_GRID = [1.25, 1.5, 2.0, 3.0, 4.0]
BLOCK_SIZES = [32, 64, 128, 256]
SEED = 0xCCC2026


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def splitmix64(idx: np.ndarray) -> np.ndarray:
    x = idx.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def block_fp16(arr2d: np.ndarray) -> np.ndarray:
    return np.frombuffer(arr2d.tobytes(), dtype="<f2").astype(np.float32)


def dequant_q8_0(payload: bytes, n: int) -> np.ndarray:
    blocks = n // Q8_0[0]
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(blocks, Q8_0[1])
    d = block_fp16(arr[:, 0:2])
    q = arr[:, 2:34].astype(np.int8).astype(np.float32)
    return (q * d[:, None]).ravel()


def dequant_q4_0(payload: bytes, n: int) -> np.ndarray:
    blocks = n // Q4_0[0]
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(blocks, Q4_0[1])
    d = block_fp16(arr[:, 0:2])
    qs = arr[:, 2:18]
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    out = np.empty((blocks, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.ravel()


def dequant_q2_K(payload: bytes, n: int) -> np.ndarray:
    blocks = n // Q2_K[0]
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(blocks, Q2_K[1])
    scales = arr[:, 0:16]
    qs = arr[:, 16:80]
    d = block_fp16(arr[:, 80:82])
    dmin = block_fp16(arr[:, 82:84])
    out = np.empty((blocks, 256), dtype=np.float32)
    for c in (0, 1):
        for j in range(4):
            q0 = ((qs[:, c * 32:c * 32 + 16] >> (2 * j)) & 3).astype(np.float32)
            q1 = ((qs[:, c * 32 + 16:c * 32 + 32] >> (2 * j)) & 3).astype(np.float32)
            sc0 = scales[:, c * 8 + 2 * j]
            sc1 = scales[:, c * 8 + 2 * j + 1]
            dl0 = d * (sc0 & 0x0F).astype(np.float32)
            ml0 = dmin * (sc0 >> 4).astype(np.float32)
            dl1 = d * (sc1 & 0x0F).astype(np.float32)
            ml1 = dmin * (sc1 >> 4).astype(np.float32)
            col0 = c * 128 + j * 32
            out[:, col0:col0 + 16] = dl0[:, None] * q0 - ml0[:, None]
            out[:, col0 + 16:col0 + 32] = dl1[:, None] * q1 - ml1[:, None]
    return out.ravel()


def dequant_q3_K(payload: bytes, n: int) -> np.ndarray:
    blocks = n // Q3_K[0]
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(blocks, Q3_K[1])
    hmask = arr[:, 0:32]
    qs = arr[:, 32:96]
    raw = arr[:, 96:108]
    d = block_fp16(arr[:, 108:110])
    aux0 = np.frombuffer(raw[:, 0:4].tobytes(), dtype="<u4").reshape(blocks)
    aux1 = np.frombuffer(raw[:, 4:8].tobytes(), dtype="<u4").reshape(blocks)
    tmp = np.frombuffer(raw[:, 8:12].tobytes(), dtype="<u4").reshape(blocks)
    a0 = (aux0 & 0x0F0F0F0F) | (((tmp >> 0) & 0x03030303) << 4)
    a1 = (aux1 & 0x0F0F0F0F) | (((tmp >> 2) & 0x03030303) << 4)
    a2 = ((aux0 >> 4) & 0x0F0F0F0F) | (((tmp >> 4) & 0x03030303) << 4)
    a3 = ((aux1 >> 4) & 0x0F0F0F0F) | (((tmp >> 6) & 0x03030303) << 4)
    u = np.stack([a0, a1, a2, a3], axis=1).astype("<u4").view(np.int8)
    u = u.reshape(blocks, 16).astype(np.float32)
    dl = d[:, None] * (u - 32.0)
    out = np.empty((blocks, 256), dtype=np.float32)
    for c in (0, 1):
        for j in range(4):
            m = 1 << (c * 4 + j)
            q0 = ((qs[:, c * 32:c * 32 + 16] >> (2 * j)) & 3).astype(np.float32)
            q1 = ((qs[:, c * 32 + 16:c * 32 + 32] >> (2 * j)) & 3).astype(np.float32)
            h0 = (hmask[:, 0:16] & m) > 0
            h1 = (hmask[:, 16:32] & m) > 0
            col0 = c * 128 + j * 32
            out[:, col0:col0 + 16] = dl[:, c * 8 + 2 * j, None] * (q0 - np.where(h0, 0.0, 4.0))
            out[:, col0 + 16:col0 + 32] = dl[:, c * 8 + 2 * j + 1, None] * (q1 - np.where(h1, 0.0, 4.0))
    return out.ravel()


def dequant_payload(type_id: int, payload: bytes, n: int) -> np.ndarray:
    table = {8: dequant_q8_0, 2: dequant_q4_0, 10: dequant_q2_K, 11: dequant_q3_K}
    return table[type_id](payload, n)


def write_minimal_gguf(path: Path, w: np.ndarray, ne0: int, ne1: int, type_id: int) -> None:
    kv = [
        ("general.architecture", "qwen3"),
        ("general.name", "ccc-qualification-tensor"),
        ("qwen3.block_count", 64),
        ("qwen3.context_length", 32768),
        ("qwen3.embedding_length", 5120),
        ("qwen3.head_count", 40),
        ("qwen3.attention.head_count_kv", 8),
        ("qwen3.feed_forward_length", 20480),
        ("qwen3.attention.layer_norm_rms_epsilon", 1e-5),
    ]

    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    blob = struct.pack("<4sIQ", b"GGUF", 3, 1) + struct.pack("<Q", len(kv))
    for k, v in kv:
        if isinstance(v, str):
            blob += s(k) + struct.pack("<I", 8) + s(v)
        elif isinstance(v, float):
            blob += s(k) + struct.pack("<I", 6) + struct.pack("<f", v)
        else:
            blob += s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
    blob += s(TENSOR_NAME)
    blob += struct.pack("<I", 2) + struct.pack("<QQ", ne0, ne1)
    blob += struct.pack("<I", type_id)
    data_offset = 0
    blob += struct.pack("<Q", data_offset)
    pad = (len(blob) + 31) // 32 * 32 - len(blob)
    path.write_bytes(blob + b"\x00" * pad + w.astype(np.float32).tobytes())


def quantize_with_llama(tool: Path, src: Path, work: Path, qtype: str) -> Path:
    out = work / f"tensor-{qtype}.gguf"
    subprocess.run([str(tool), str(src), str(out), qtype], check=True, capture_output=True)
    return out


def err_stats(err: np.ndarray, w_ref: np.ndarray) -> dict:
    a = np.abs(err)
    q = lambda p: float(np.quantile(a, p))
    return {
        "weight_mae": float(a.mean()),
        "weight_rmse": float(math.sqrt((err.astype(np.float64) ** 2).mean())),
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(w_ref)),
        "p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99),
        "p99_9": q(0.999), "max": float(a.max()),
    }


def residual_stats(r: np.ndarray, raw: np.ndarray) -> dict:
    a = np.abs(r)
    q = lambda p: float(np.quantile(a, p))
    mean = float(r.mean())
    std = float(r.std())
    skew = float(((r - mean) ** 3).mean() / (std ** 3 + 1e-30))
    kurt = float(((r - mean) ** 4).mean() / (std ** 4 + 1e-30))
    return {
        "residual_rms": float(np.sqrt((r.astype(np.float64) ** 2).mean())),
        "residual_variance": float((r.astype(np.float64) ** 2).mean()),
        "residual_mae": float(a.mean()),
        "residual_p50": q(0.50), "residual_p90": q(0.90), "residual_p95": q(0.95),
        "residual_p99": q(0.99), "residual_p99_9": q(0.999), "residual_max": float(a.max()),
        "residual_skew": skew, "residual_kurtosis": kurt,
        "residual_rms_over_raw_rms": float(np.sqrt((r ** 2).mean()) / np.sqrt((raw ** 2).mean())),
        "residual_energy_over_raw_energy": float((r ** 2).sum() / (raw ** 2).sum()),
    }


def hist_entropy(r: np.ndarray, bins: int = 1024) -> float:
    lo, hi = np.quantile(r, [0.001, 0.999])
    if hi - lo <= 0:
        return 0.0
    h, _ = np.histogram(r, bins=bins, range=(lo, hi))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() - np.log2((hi - lo) / bins))


def lloyd_fit(r: np.ndarray, levels: int = 8, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(7)
    lo, hi = np.quantile(r, [0.02, 0.98])
    L = np.linspace(lo, hi, levels)
    for _ in range(iters):
        assign = np.argmin(np.abs(r[:, None] - L[None, :]), axis=1)
        for k in range(levels):
            m = assign == k
            if m.any():
                L[k] = r[m].mean()
    return L


def assign_codes(r: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(r[:, None] - levels[None, :]), axis=1)


def pow_levels(R: float, gamma: float, special: tuple | None = None, no_zero: bool = False) -> np.ndarray:
    mags = R * (np.arange(1, 5) / 4.0) ** gamma if no_zero else R * (np.array([1, 2, 3]) / 3.0) ** gamma
    L = np.concatenate([[0.0], mags, -mags])
    if special is not None:
        side, T = special
        L = np.concatenate([L, [T * R if side == "pos" else -T * R]])
    return L


def pick_R(res: np.ndarray, gamma: float, special: tuple | None, no_zero: bool,
           val: np.ndarray) -> tuple[float, str, np.ndarray]:
    """Choose R on validation; returns (R, R_source_name, levels)."""
    sigma = float(res.std())
    a = np.abs(res)
    cands = {"2sigma": 2 * sigma, "3sigma": 3 * sigma, "4sigma": 4 * sigma,
             "q99": float(np.quantile(a, 0.99)), "q99.9": float(np.quantile(a, 0.999))}
    best, best_R, best_rn, best_L = None, None, None, None
    for rn, R in cands.items():
        L = pow_levels(R, gamma, special, no_zero)
        codes = assign_codes(val, L)
        mse = float(((val - L[codes]) ** 2).mean())
        if best is None or mse < best:
            best, best_R, best_rn, best_L = mse, R, rn, L
    if best_R is None or best_rn is None or best_L is None:
        raise RuntimeError("degenerate R selection")
    return best_R, best_rn, best_L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=ROOT / "research-models/Qwen3-32B-Q8_0.gguf")
    ap.add_argument("--llama-quantize", type=Path,
                    default=Path("/tmp/ccc-llama-pinned/build/bin/llama-quantize"))
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "benchmark-results/ccc-structured-baseline-qualification")
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip the 34.8 GB sha256 pass (already verified once)")
    args = ap.parse_args()
    out = args.output_dir.resolve()
    work = Path("/tmp/opencode/ccc-qualification-work")
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    prov = {"artifact": str(args.gguf), "expected_sha256": GGUF_SHA256,
            "llama_cpp_commit": PINNED_COMMIT, "llama_quantize": str(args.llama_quantize),
            "tensor": TENSOR_NAME,
            "ground_truth": "Q8_0 reconstructed weights (Q8-relative, NOT BF16/F32)"}
    if not args.skip_hash:
        prov["sha256"] = sha256_file(args.gguf)
        if prov["sha256"] != GGUF_SHA256:
            raise SystemExit("artifact hash mismatch")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")

    art = parse(args.gguf)
    t = [x for x in art.tensors if x.name == TENSOR_NAME][0]
    assert t.type_name == "Q8_0"
    ne0, ne1 = t.shape[0], t.shape[1]
    if t.elements is None or t.payload_size is None:
        raise SystemExit("missing tensor size fields")
    n = int(t.elements)
    with args.gguf.open("rb") as f:
        f.seek(t.absolute_start)
        payload = f.read(t.payload_size)
    W = dequant_q8_0(payload, n).reshape(ne1, ne0).astype(np.float64)
    rows, cols = ne1, ne0  # rows = output dim (outer), columns = input dim (inner)
    N = rows * cols

    h = splitmix64(np.arange(n, dtype=np.uint64)) % np.uint64(100)
    mask = np.where(h < 50, 0, np.where(h < 75, 1, 2)).astype(np.int8).reshape(rows, cols)
    fit, val, test = mask == 0, mask == 1, mask == 2
    n_fit, n_val, n_test = int(fit.sum()), int(val.sum()), int(test.sum())
    Wf, Wv, Wt = W[fit], W[val], W[test]

    # ---------- baselines (fitted on FIT only) ----------
    g_mean = float(Wf.mean())
    g_med = float(np.median(Wf))
    row_mean = np.array([W[r][fit[r]].mean() for r in range(rows)])
    row_med = np.array([np.median(W[r][fit[r]]) for r in range(rows)])
    col_mean = np.array([W[:, c][fit[:, c]].mean() for c in range(cols)])
    col_med = np.array([np.median(W[:, c][fit[:, c]]) for c in range(cols)])
    row_dev, col_dev = row_mean - g_mean, col_mean - g_mean

    B = {
        "B0": np.zeros((rows, cols)),
        "B1": np.full((rows, cols), g_mean),
        "B2": np.full((rows, cols), g_med),
        "B3": row_mean[:, None] * np.ones((1, cols)),
        "B4": col_mean[None, :] * np.ones((rows, 1)),
        "B5": g_mean + row_dev[:, None] + col_dev[None, :],
    }
    meta_bits = {"B0": 0, "B1": 32, "B2": 32, "B3": rows * 32,
                 "B4": cols * 32, "B5": (rows + cols + 1) * 32}

    # ---------- predictor quality (TEST) ----------
    pred_rows = []
    base = {"baseline": "raw", "metadata_bpw": 0.0, "residual_entropy_est": hist_entropy(Wt)}
    base.update(residual_stats(Wt, Wt))
    pred_rows.append(base)
    for name in B:
        r = Wt - B[name][test]
        st = {"baseline": name, "metadata_bpw": meta_bits[name] / N,
              "residual_entropy_est": hist_entropy(r)}
        st.update(residual_stats(r, Wt))
        pred_rows.append(st)
    for name, bb in (("B3_med", row_med[:, None] * np.ones((1, cols))),
                     ("B4_med", col_med[None, :] * np.ones((rows, 1)))):
        r = Wt - bb[test]
        st = {"baseline": name, "metadata_bpw": meta_bits[name.replace("_med", "")] / N,
              "residual_entropy_est": hist_entropy(r)}
        st.update(residual_stats(r, Wt))
        pred_rows.append(st)
    with (out / "predictor-quality.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(pred_rows[0]), lineterminator="\n")
        wcsv.writeheader()
        wcsv.writerows(pred_rows)

    # ---------- alphabet fit + evaluate ----------
    rows_out = []
    selected: dict[tuple, np.ndarray] = {}

    def evaluate(bname: str, aname: str, L: np.ndarray, scale_scope: str,
                 metadata_bytes: float, sel: dict) -> dict:
        rt = Wt - B[bname][test]
        codes = assign_codes(rt, L)
        w_hat = B[bname][test] + L[codes]
        st = err_stats(w_hat - Wt, Wt)
        uniq, cnts = np.unique(codes, return_counts=True)
        mag_max = float(np.abs(L).max())
        clip = int((np.abs(rt) > mag_max).sum())
        zero_occ = int((L[codes] == 0).sum())
        special_occ = int((np.abs(L[codes]) == mag_max).sum()) if sel.get("special") else 0
        meta_bpw = (meta_bits[bname] + metadata_bytes) / N
        return {
            "candidate": f"{bname}_{aname}", "baseline": bname, "alphabet": aname,
            "scale_scope": scale_scope, "true_bpw": 3 + meta_bpw, "metadata_bpw": meta_bpw,
            "code_bpw": 3, "weight_mae": st["weight_mae"], "weight_rmse": st["weight_rmse"],
            "rel_l2": st["rel_l2"], "p99": st["p99"], "p99_9": st["p99_9"], "max": st["max"],
            "clipping": clip, "clipping_pct": 100.0 * clip / n_test,
            "zero_state_occupancy_pct": 100.0 * zero_occ / n_test,
            "special_state_occupancy_pct": 100.0 * special_occ / n_test,
            "per_code_occupancy": json.dumps({int(u): int(c) for u, c in zip(uniq, cnts)}),
            "selected": json.dumps(sel, sort_keys=True), "direct_apply": "UNPROVEN",
            "wx_rel_l2_mean": None, "cosine_mean": None, "wx_max_err_mean": None,
        }

    for bname in B:
        rf, rv = Wf - B[bname][fit], Wv - B[bname][val]
        # G0 free 8-state (mandatory upper-bound control)
        L = lloyd_fit(rf, 8)
        rows_out.append(evaluate(bname, "G0_free8", L, "global", 8 * 32.0,
                                 {"alpha": "lloyd8", "levels": L.tolist()}))
        selected[(bname, "G0_free8")] = L
        # G1 mirrored linear
        R, rn, L = pick_R(rf, 1.0, None, False, rv)
        rows_out.append(evaluate(bname, "G1_linear", L, "global", 4.0,
                                 {"alpha": "linear", "R": R, "R_source": rn, "gamma": 1.0}))
        selected[(bname, "G1_linear")] = L
        # G2 mirrored power over gamma grid
        for gamma in GAMMA_GRID:
            R, rn, L = pick_R(rf, gamma, None, False, rv)
            aname = f"G2_power_g{gamma}"
            rows_out.append(evaluate(bname, aname, L, "global", 8.0,
                                     {"alpha": "power", "R": R, "R_source": rn, "gamma": gamma}))
            selected[(bname, aname)] = L
        # G3 zero + three mirrored + special tail
        side = "pos" if rf[rf > 0].sum() > -rf[rf < 0].sum() else "neg"
        for gamma in GAMMA_GRID:
            best = None
            for T in TAIL_GRID:
                R, rn, L = pick_R(rf, gamma, (side, T), False, rv)
                codes = assign_codes(rv, L)
                mse = float(((rv - L[codes]) ** 2).mean())
                if best is None or mse < best[0]:
                    best = (mse, R, rn, T, L)
            if best is None:
                raise RuntimeError("degenerate G3 fit")
            _, R, rn, T, L = best
            aname = f"G3_tail_g{gamma}"
            rows_out.append(evaluate(bname, aname, L, "global", 12.0,
                                     {"alpha": "power+special", "R": R, "R_source": rn,
                                      "gamma": gamma, "T": T, "side": side, "special": True}))
            selected[(bname, aname)] = L
        # G4 no-zero mirrored 8-state (control)
        R, rn, L = pick_R(rf, GAMMA_GRID[3], None, True, rv)
        rows_out.append(evaluate(bname, "G4_nozero", L, "global", 8.0,
                                 {"alpha": "power-nozero", "R": R, "R_source": rn,
                                  "gamma": GAMMA_GRID[3]}))
        selected[(bname, "G4_nozero")] = L

    # ---------- per-block scale (global curve + per-block R_b) ----------
    best_g2 = min((r for r in rows_out if "G2_power" in r["alphabet"]), key=lambda r: r["weight_rmse"])
    bb, ba = best_g2["baseline"], best_g2["alphabet"]
    gamma = json.loads(best_g2["selected"])["gamma"]
    r_blocks: dict[int, np.ndarray] = {}
    for bs in BLOCK_SIZES:
        nb = cols // bs
        r_b = np.zeros((rows, nb))
        for r in range(rows):
            row_fit = fit[r]
            for b in range(nb):
                seg = slice(b * bs, (b + 1) * bs)
                rfit = W[r][seg][row_fit[seg]] - B[bb][r][seg][row_fit[seg]]
                if rfit.size == 0:
                    r_b[r, b] = 1.0
                    continue
                a = np.abs(rfit)
                cands = [float(np.quantile(a, q)) for q in (0.75, 0.90, 0.95, 0.99)] + [float(a.max())]
                best_s, best_mse = cands[0], None
                rval = W[r][seg][val[r][seg]] - B[bb][r][seg][val[r][seg]]
                if rval.size == 0:
                    r_b[r, b] = max(best_s, 1e-12)
                    continue
                for s in cands:
                    Lb = pow_levels(s, gamma, None, False)
                    c = assign_codes(rval, Lb)
                    mse = float(((rval - Lb[c]) ** 2).mean())
                    if best_mse is None or mse < best_mse:
                        best_mse, best_s = mse, s
                r_b[r, b] = max(best_s, 1e-12)
        r_blocks[bs] = r_b
        # evaluate on TEST
        rt = Wt - B[bb][test]
        pos = np.where(test)
        rb_flat = r_b[pos[0], pos[1] // bs]
        mags = np.stack([rb_flat * ((k / 3.0) ** gamma) for k in (1, 2, 3)], axis=1)
        Lb = np.concatenate([np.zeros((n_test, 1)), mags, -mags], axis=1)
        codes = np.argmin(np.abs(rt[:, None] - Lb), axis=1)
        w_hat = B[bb][test] + np.take_along_axis(Lb, codes[:, None], axis=1)[:, 0]
        st = err_stats(w_hat - Wt, Wt)
        mag_max = np.abs(Lb).max()
        clip = int((np.abs(rt) > mag_max).sum())
        meta_bpw = (meta_bits[bb] + 4 + 16.0 * rows * nb) / N
        rows_out.append({
            "candidate": f"{bb}_{ba}_perblock{bs}", "baseline": bb, "alphabet": ba,
            "scale_scope": f"per_block_{bs}", "true_bpw": 3 + meta_bpw, "metadata_bpw": meta_bpw,
            "code_bpw": 3, "weight_mae": st["weight_mae"], "weight_rmse": st["weight_rmse"],
            "rel_l2": st["rel_l2"], "p99": st["p99"], "p99_9": st["p99_9"], "max": st["max"],
            "clipping": clip, "clipping_pct": 100.0 * clip / n_test,
            "zero_state_occupancy_pct": None, "special_state_occupancy_pct": None,
            "per_code_occupancy": None,
            "selected": json.dumps({"alpha": "power+perblockR", "gamma": gamma, "block": bs,
                                    "R_b": "per-block fit on FIT samples"}),
            "direct_apply": "UNPROVEN", "wx_rel_l2_mean": None, "cosine_mean": None,
            "wx_max_err_mean": None,
        })

    # ---------- canonical controls (pinned llama.cpp) ----------
    src_gguf = work / "tensor-f32.gguf"
    write_minimal_gguf(src_gguf, W.astype(np.float32), ne0, ne1, 0)
    for qname, (bsz, bbytes, tid, code_bits) in (("Q8_0", Q8_0), ("Q4_0", Q4_0),
                                                 ("Q2_K", Q2_K), ("Q3_K", Q3_K)):
        qfile = quantize_with_llama(args.llama_quantize, src_gguf, work, qname)
        qart = parse(qfile)
        qt = [x for x in qart.tensors if x.name == TENSOR_NAME][0]
        with qfile.open("rb") as f:
            f.seek(qt.absolute_start)
            qpayload = f.read(qt.payload_size)
        Wq = dequant_payload(tid, qpayload, n).reshape(ne1, ne0).astype(np.float64)
        st = err_stats(Wq[test] - Wt, Wt)
        true_bpw = 8.0 * bbytes / bsz
        meta_bpw = true_bpw - code_bits
        row = {
            "candidate": f"canonical_{qname}", "baseline": "none", "alphabet": "ggml",
            "scale_scope": "block", "true_bpw": true_bpw, "metadata_bpw": meta_bpw,
            "code_bpw": code_bits, "weight_mae": st["weight_mae"], "weight_rmse": st["weight_rmse"],
            "rel_l2": st["rel_l2"], "p99": st["p99"], "p99_9": st["p99_9"], "max": st["max"],
            "clipping": None, "clipping_pct": None, "zero_state_occupancy_pct": None,
            "special_state_occupancy_pct": None, "per_code_occupancy": None,
            "selected": json.dumps({"type": qname, "bytes_per_block": bbytes,
                                    "weights_per_block": bsz}),
            "direct_apply": "MEASURED_BY_GGML", "wx_rel_l2_mean": None,
            "cosine_mean": None, "wx_max_err_mean": None,
        }
        if qname == "Q8_0":
            row["q8_roundtrip_fraction_identical"] = float((np.abs(Wq - W) < 1e-6).mean())
        rows_out.append(row)
    # simplified uniform Q3 — SIMPLIFIED CONTROL (continuity with prior exploratory work)
    blk = W.reshape(rows, cols // 32, 32)
    dm = (np.abs(blk).max(axis=2) / 4.0).astype(np.float32)
    dm[dm == 0] = 1e-9
    qq = np.clip(np.round(blk / dm[:, :, None]), -4, 3)
    Wq_simpl = (qq * dm[:, :, None]).reshape(rows, cols)
    st = err_stats(Wq_simpl[test] - Wt, Wt)
    rows_out.append({
        "candidate": "simplified_uniform_Q3", "baseline": "none", "alphabet": "uniform3",
        "scale_scope": "block32", "true_bpw": 3 + 16.0 / 32, "metadata_bpw": 16.0 / 32,
        "code_bpw": 3, "weight_mae": st["weight_mae"], "weight_rmse": st["weight_rmse"],
        "rel_l2": st["rel_l2"], "p99": st["p99"], "p99_9": st["p99_9"], "max": st["max"],
        "clipping": None, "clipping_pct": None, "zero_state_occupancy_pct": None,
        "special_state_occupancy_pct": None, "per_code_occupancy": None,
        "selected": json.dumps({"note": "SIMPLIFIED CONTROL — uniform 3-bit, per-32 fp16 scale"}),
        "direct_apply": "UNPROVEN", "wx_rel_l2_mean": None, "cosine_mean": None,
        "wx_max_err_mean": None,
    })

    # ---------- W*x probes ----------
    rng = np.random.default_rng(12345)
    X = rng.standard_normal((3, cols))
    y_ref = W @ X.T
    best_g2_b5 = min((r for r in rows_out if r["baseline"] == "B5" and "G2_power" in r["alphabet"]),
                     key=lambda r: r["weight_rmse"])
    best_g3_b5 = min((r for r in rows_out if r["baseline"] == "B5" and "G3_tail" in r["alphabet"]),
                     key=lambda r: r["weight_rmse"])
    best_pb = min((r for r in rows_out if "perblock" in r["candidate"]),
                  key=lambda r: r["weight_rmse"])
    wx_candidates = [best_g2_b5["candidate"], best_g3_b5["candidate"],
                     "canonical_Q8_0", "canonical_Q4_0", "canonical_Q3_K",
                     "simplified_uniform_Q3", best_pb["candidate"]]
    wx_rows = []
    for cand in wx_candidates:
        row = [r for r in rows_out if r["candidate"] == cand][0]
        if cand.startswith("canonical_"):
            qname = cand.replace("canonical_", "")
            qfile = work / f"tensor-{qname}.gguf"
            qart = parse(qfile)
            qt = [x for x in qart.tensors if x.name == TENSOR_NAME][0]
            with qfile.open("rb") as f:
                f.seek(qt.absolute_start)
                qpayload = f.read(qt.payload_size)
            Wc = dequant_payload(qt.type_id, qpayload, n).reshape(ne1, ne0).astype(np.float64)
        elif cand == "simplified_uniform_Q3":
            Wc = Wq_simpl.astype(np.float64)
        elif "perblock" in cand:
            bs = int(cand.split("perblock")[1])
            r_b = r_blocks[bs]
            gamma = json.loads(row["selected"])["gamma"]
            ri, ci = np.indices((rows, cols))
            rb_flat = r_b[ri, ci // bs]
            mags = np.stack([rb_flat * ((k / 3.0) ** gamma) for k in (1, 2, 3)], axis=-1)
            Lb = np.concatenate([np.zeros((rows, cols, 1)), mags, -mags], axis=-1)
            codes = np.argmin(np.abs((W - B[bb])[..., None] - Lb), axis=-1)
            Wc = B[bb] + np.take_along_axis(Lb, codes[..., None], axis=-1)[..., 0]
        else:
            bname, aname = row["baseline"], row["alphabet"]
            L = selected[(bname, aname)]
            codes = assign_codes((W - B[bname]).ravel(), L)
            Wc = B[bname] + L[codes].reshape(rows, cols)
        for k in range(3):
            y_c = Wc @ X[k]
            rel = float(np.linalg.norm(y_c - y_ref[:, k]) / np.linalg.norm(y_ref[:, k]))
            cos = float(np.dot(y_c, y_ref[:, k]) / (np.linalg.norm(y_c) * np.linalg.norm(y_ref[:, k]) + 1e-30))
            wx_rows.append({"candidate": cand, "probe": k, "wx_rel_l2": rel, "cosine": cos,
                            "wx_max_err": float(np.abs(y_c - y_ref[:, k]).max())})

    # B3/B4/B5 algebraic separation invariant
    inv_rows = []
    for name in ("B3", "B4", "B5"):
        L = selected[(name, "G2_power_g1.35")]
        codes = assign_codes((W - B[name]).ravel(), L)
        C = L[codes].reshape(rows, cols)
        for k in range(3):
            x = X[k]
            dense = (B[name] + C) @ x
            if name == "B5":
                sep = (C @ x) + (g_mean + row_dev) * float(x.sum()) + float(col_dev @ x)
            elif name == "B3":
                sep = (C @ x) + row_mean * float(x.sum())
            else:
                sep = (C @ x) + float(col_mean @ x)
            max_abs = float(np.abs(dense - sep).max())
            max_rel = float(np.abs(dense - sep).max() / (np.abs(dense).max() + 1e-30))
            inv_rows.append({"baseline": name, "probe": k, "max_abs_diff": max_abs,
                             "max_rel_diff": max_rel})
    (out / "wx-probes.json").write_text(json.dumps(
        {"y_ref": "W_Q8 @ x (deterministic N(0,1) activations, 3 probes)",
         "probes": wx_rows, "separation_invariant": inv_rows}, indent=2) + "\n")

    for r in rows_out:
        wx = [w for w in wx_rows if w["candidate"] == r["candidate"]]
        if wx:
            r["wx_rel_l2_mean"] = float(np.mean([w["wx_rel_l2"] for w in wx]))
            r["cosine_mean"] = float(np.mean([w["cosine"] for w in wx]))
            r["wx_max_err_mean"] = float(np.mean([w["wx_max_err"] for w in wx]))

    # ---------- outputs ----------
    fieldnames = list(rows_out[0])
    for r in rows_out:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with (out / "ccc_structured_baseline_qualification.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        wcsv.writeheader()
        wcsv.writerows(rows_out)
    low = [r for r in rows_out if r["true_bpw"] <= 3.1]
    mid = [r for r in rows_out if r["true_bpw"] <= 3.7]
    summary = {
        "tensor": {"name": TENSOR_NAME, "gguf_shape": list(t.shape), "ne0_inner": int(ne0),
                   "ne1_outer": int(ne1), "rows": int(rows), "columns": int(cols),
                   "elements": int(n), "ggml_type": t.type_name,
                   "payload_bytes": int(t.payload_size)},
        "split": {"fit": n_fit, "val": n_val, "test": n_test,
                  "method": "deterministic splitmix64(row,col) — every row contributes FIT samples"},
        "n_candidates": len(rows_out),
        "best_by_rmse_below_3_1bpw": sorted(
            [{"candidate": r["candidate"], "true_bpw": r["true_bpw"], "weight_rmse": r["weight_rmse"],
              "p99_9": r["p99_9"]} for r in low], key=lambda r: r["weight_rmse"])[:6],
        "best_by_rmse_below_3_7bpw": sorted(
            [{"candidate": r["candidate"], "true_bpw": r["true_bpw"], "weight_rmse": r["weight_rmse"],
              "p99_9": r["p99_9"]} for r in mid], key=lambda r: r["weight_rmse"])[:8],
        "canonical": {r["candidate"]: {"true_bpw": r["true_bpw"], "weight_rmse": r["weight_rmse"],
                                       "p99_9": r["p99_9"]} for r in rows_out
                      if r["candidate"].startswith("canonical")},
        "separation_invariant_max_rel_diff": max((i["max_rel_diff"] for i in inv_rows), default=None),
        "direct_apply": "DIRECT_APPLY_UNPROVEN — algebraic separation verified numerically; no kernel measured",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(
        {"python": sys.version.split()[0], "numpy": np.__version__,
         "llama_cpp_commit": PINNED_COMMIT, "worktree": "/tmp/ccc-llama-pinned",
         "patches_applied": ["0001-user-metadata-tensor-source.patch",
                             "0002-source-neutral-model-source.patch"],
         "artifact": prov.get("sha256"), "ground_truth": prov["ground_truth"]}, indent=2) + "\n")
    print(f"PASS — tensor {TENSOR_NAME} shape={t.shape} n={n} rows={rows} cols={cols}; "
          f"candidates={len(rows_out)}; "
          f"best<3.1bpw={summary['best_by_rmse_below_3_1bpw'][0] if low else None}; "
          f"invariant_max_rel={summary['separation_invariant_max_rel_diff']}")


if __name__ == "__main__":
    main()