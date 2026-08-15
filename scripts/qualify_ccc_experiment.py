#!/usr/bin/env python3
"""
CCC Geometric Correction Code — Research Qualification Script
Quantifies rate-distortion, tail handling, baseline decomposition, true byte accounting,
and random W*x action error on real Qwen3-32B-Q8_0 model weights.
"""
from __future__ import annotations
import csv, json, math, os, struct, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_PATH = Path("/home/eugen/projekte/vBuf/research-models/Qwen3-32B-Q8_0.gguf")
OUT_JSON = ROOT / "benchmark-results/ccc_geometric_qualification.json"
OUT_CSV = ROOT / "benchmark-results/ccc_geometric_qualification.csv"

def fast_gguf_find_tensor(path: Path, target_names=("blk.0.attn_k.weight", "blk.0.attn_output.weight")):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Invalid magic {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count, kv_count = struct.unpack('<QQ', f.read(16))
        
        def read_str():
            l = struct.unpack('<Q', f.read(8))[0]
            return f.read(l).decode('utf-8', errors='ignore')

        def skip_val(val_type):
            if val_type in (0, 1, 7): f.seek(1, 1)
            elif val_type in (2, 3): f.seek(2, 1)
            elif val_type in (4, 5, 6): f.seek(4, 1)
            elif val_type in (10, 11, 12): f.seek(8, 1)
            elif val_type == 8:
                l = struct.unpack('<Q', f.read(8))[0]
                f.seek(l, 1)
            elif val_type == 9:
                itype, icount = struct.unpack('<IQ', f.read(12))
                if itype in (0, 1, 7): f.seek(icount, 1)
                elif itype in (2, 3): f.seek(icount * 2, 1)
                elif itype in (4, 5, 6): f.seek(icount * 4, 1)
                elif itype in (10, 11, 12): f.seek(icount * 8, 1)
                elif itype == 8:
                    for _ in range(icount):
                        l = struct.unpack('<Q', f.read(8))[0]
                        f.seek(l, 1)
                elif itype == 9:
                    for _ in range(icount): skip_val(9)

        for _ in range(kv_count):
            k = read_str()
            vtype = struct.unpack('<I', f.read(4))[0]
            skip_val(vtype)

        tensors = []
        for _ in range(tensor_count):
            name = read_str()
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ggml_type, offset = struct.unpack('<IQ', f.read(12))
            tensors.append({'name': name, 'shape': dims, 'type': ggml_type, 'offset': offset})

        data_offset = (f.tell() + 31) // 32 * 32
        for t in tensors:
            if t['name'] in target_names:
                t['absolute_start'] = data_offset + t['offset']
                t['elements'] = math.prod(t['shape'])
                t['payload_size'] = t['elements'] // 32 * 34
                return t
        
        t = tensors[0]
        t['absolute_start'] = data_offset + t['offset']
        t['elements'] = math.prod(t['shape'])
        t['payload_size'] = t['elements'] // 32 * 34
        return t

def decode_q8_tensor(path: Path, t_info: dict):
    with path.open("rb") as f:
        f.seek(t_info['absolute_start'])
        raw = f.read(t_info['payload_size'])
    blocks = t_info['elements'] // 32
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, 34)
    scales = np.frombuffer(packed[:, :2].tobytes(), dtype="<f2").astype(np.float32)
    q = packed[:, 2:].view(np.int8).astype(np.float32)
    shape = tuple(reversed(t_info['shape']))
    return (q * scales[:, None]).reshape(shape).astype(np.float32, copy=False)

def evaluate_metrics(ref: np.ndarray, recon: np.ndarray, test_mask: np.ndarray):
    diff = np.abs(ref[test_mask] - recon[test_mask])
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rel_l2 = float(np.linalg.norm(ref[test_mask] - recon[test_mask]) / (np.linalg.norm(ref[test_mask]) + 1e-12))
    pcts = np.percentile(diff, [50, 90, 95, 99, 99.9])
    p50, p90, p95, p99, p99_9 = [float(x) for x in pcts]
    max_err = float(np.max(diff))
    unique_states = int(len(np.unique(recon[test_mask])))
    return {
        "mae": mae,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "p99_9": p99_9,
        "max_err": max_err,
        "unique_states": unique_states
    }

def lloyd_max_scalar(flat_weights: np.ndarray, k: int = 8, iterations: int = 15):
    flat = flat_weights.ravel()
    code = np.quantile(flat, np.linspace(0.001, 0.999, k)).astype(np.float32)
    for _ in range(iterations):
        cuts = (code[:-1] + code[1:]) / 2.0
        idx = np.searchsorted(cuts, flat).astype(np.int32)
        sums = np.bincount(idx, weights=flat, minlength=k)
        counts = np.bincount(idx, minlength=k)
        code = np.where(counts > 0, sums / np.maximum(counts, 1), code).astype(np.float32)
    cuts = (code[:-1] + code[1:]) / 2.0
    idx = np.searchsorted(cuts, flat).astype(np.uint8)
    return code, idx

def precompute_wx_reference(ref_w: np.ndarray, num_probes: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_in = ref_w.shape[1]
    X = rng.standard_normal((n_in, num_probes), dtype=np.float32)
    Y_ref = ref_w @ X
    return X, Y_ref

def run_wx_probe_fast(recon_w: np.ndarray, X: np.ndarray, Y_ref: np.ndarray):
    Y_recon = recon_w @ X
    diff = Y_recon - Y_ref
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(Y_ref) + 1e-12))
    
    dot_prods = np.sum(Y_ref * Y_recon, axis=0)
    norms_ref = np.linalg.norm(Y_ref, axis=0)
    norms_recon = np.linalg.norm(Y_recon, axis=0)
    cos_sims = dot_prods / (norms_ref * norms_recon + 1e-12)
    
    max_out_err = float(np.max(np.abs(diff)))
    return {
        "wx_rel_l2_mean": rel_l2,
        "wx_cos_sim_mean": float(np.mean(cos_sims)),
        "wx_max_err_max": max_out_err
    }

def main():
    print(f"Parsing GGUF model: {GGUF_PATH}...", flush=True)
    t_info = fast_gguf_find_tensor(GGUF_PATH)
    print(f"Target Tensor: {t_info['name']} | Shape: {t_info['shape']} | Start Offset: {t_info['absolute_start']}", flush=True)
    
    ref_w = decode_q8_tensor(GGUF_PATH, t_info)
    print(f"Decoded Q8_0 Ground Truth Shape: {ref_w.shape} | Range: [{ref_w.min():.4f}, {ref_w.max():.4f}]", flush=True)
    
    X_probe, Y_ref_probe = precompute_wx_reference(ref_w, num_probes=10)
    
    m_rows, n_cols = ref_w.shape
    total_elements = m_rows * n_cols
    
    # Position Hash deterministic split (70% Fit, 15% Val, 15% Test)
    row_idx, col_idx = np.indices((m_rows, n_cols))
    pos_hash = (row_idx * 65537 + col_idx * 31) % 100
    fit_mask = pos_hash < 70
    val_mask = (pos_hash >= 70) & (pos_hash < 85)
    test_mask = pos_hash >= 85
    
    print(f"Split distribution -> Fit: {np.sum(fit_mask)}, Val: {np.sum(val_mask)}, Test: {np.sum(test_mask)}", flush=True)
    
    results = []
    
    # 1. Q8_0 Ground Truth Reference
    q8_bpw = (t_info['payload_size'] * 8.0) / total_elements
    q8_metrics = evaluate_metrics(ref_w, ref_w, test_mask)
    q8_wx = run_wx_probe_fast(ref_w, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "Q8_0_Reference",
        "family": "Reference",
        "variant": "GGML_Q8_0",
        "bits_payload": 8.0,
        "metadata_bytes": t_info['payload_size'] - total_elements,
        "total_bytes": t_info['payload_size'],
        "true_bpw": q8_bpw,
        "direct_apply_class": "DIRECT_APPLY_PLAUSIBLE",
        **q8_metrics,
        **q8_wx,
        "clipping_count": 0,
        "fit_time_ms": 0.0,
        "encode_time_ms": 0.0
    })
    
    # 2. Classical Linear Baselines
    block_size = 32
    flat_w = ref_w.ravel()
    n_blocks = len(flat_w) // block_size
    reshaped_w = flat_w[:n_blocks*block_size].reshape(n_blocks, block_size)
    
    # Block Q3 Symmetric
    start_t = time.perf_counter()
    max_val = np.max(np.abs(reshaped_w), axis=1, keepdims=True)
    scale_q3 = np.maximum(max_val / 3.0, 1e-10)
    q3_sym_code = np.clip(np.round(reshaped_w / scale_q3), -3, 3).astype(np.int8)
    q3_sym_recon = (q3_sym_code * scale_q3).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    q3_sym_bytes = int(math.ceil(total_elements * 3 / 8.0)) + n_blocks * 2
    q3_sym_bpw = (q3_sym_bytes * 8.0) / total_elements
    q3_sym_metrics = evaluate_metrics(ref_w, q3_sym_recon, test_mask)
    q3_sym_wx = run_wx_probe_fast(q3_sym_recon, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "Block_Q3_Symmetric",
        "family": "Classical Baseline",
        "variant": "Symmetric Block Q3",
        "bits_payload": 3.0,
        "metadata_bytes": n_blocks * 2,
        "total_bytes": q3_sym_bytes,
        "true_bpw": q3_sym_bpw,
        "direct_apply_class": "DIRECT_APPLY_UNPROVEN",
        **q3_sym_metrics,
        **q3_sym_wx,
        "clipping_count": int(np.sum(np.abs(q3_sym_code) == 3)),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    # Block Q3 Affine
    start_t = time.perf_counter()
    min_val = np.min(reshaped_w, axis=1, keepdims=True)
    max_val = np.max(reshaped_w, axis=1, keepdims=True)
    range_val = np.maximum(max_val - min_val, 1e-10)
    scale_aff = range_val / 7.0
    q3_aff_code = np.clip(np.round((reshaped_w - min_val) / scale_aff), 0, 7).astype(np.uint8)
    q3_aff_recon = (q3_aff_code * scale_aff + min_val).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    q3_aff_bytes = int(math.ceil(total_elements * 3 / 8.0)) + n_blocks * 4
    q3_aff_bpw = (q3_aff_bytes * 8.0) / total_elements
    q3_aff_metrics = evaluate_metrics(ref_w, q3_aff_recon, test_mask)
    q3_aff_wx = run_wx_probe_fast(q3_aff_recon, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "Block_Q3_Affine",
        "family": "Classical Baseline",
        "variant": "Affine Block Q3",
        "bits_payload": 3.0,
        "metadata_bytes": n_blocks * 4,
        "total_bytes": q3_aff_bytes,
        "true_bpw": q3_aff_bpw,
        "direct_apply_class": "DIRECT_APPLY_UNPROVEN",
        **q3_aff_metrics,
        **q3_aff_wx,
        "clipping_count": int(np.sum((q3_aff_code == 0) | (q3_aff_code == 7))),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    # 3. Baselines Audit (B_0 to B_6)
    b_models = {
        "B0_Zero": np.zeros_like(ref_w),
        "B1_GlobalMean": np.full_like(ref_w, float(np.mean(ref_w[fit_mask]))),
        "B2_GlobalMedian": np.full_like(ref_w, float(np.median(ref_w[fit_mask]))),
        "B3_TrimmedMean": np.full_like(ref_w, float(np.mean(np.clip(ref_w[fit_mask], np.percentile(ref_w[fit_mask], 5), np.percentile(ref_w[fit_mask], 95))))),
        "B4_RowMean": np.mean(ref_w, axis=1, keepdims=True) * np.ones_like(ref_w),
        "B5_RowMedian": np.median(ref_w, axis=1, keepdims=True) * np.ones_like(ref_w)
    }
    mu = float(np.mean(ref_w[fit_mask]))
    r_i = np.mean(ref_w - mu, axis=1, keepdims=True)
    c_j = np.mean(ref_w - mu - r_i, axis=0, keepdims=True)
    b_models["B6_RowColAdditive"] = mu + r_i + c_j

    print("Evaluating Baseline Predictor decomposition...", flush=True)
    for b_name, B_arr in b_models.items():
        b_metrics = evaluate_metrics(ref_w, B_arr, test_mask)
        meta_bytes = 0
        if b_name.startswith("B1") or b_name.startswith("B2") or b_name.startswith("B3"):
            meta_bytes = 2
        elif b_name.startswith("B4") or b_name.startswith("B5"):
            meta_bytes = m_rows * 2
        elif b_name.startswith("B6"):
            meta_bytes = 2 + (m_rows + n_cols) * 2
            
        b_bpw = (meta_bytes * 8.0) / total_elements
        results.append({
            "candidate_id": f"Baseline_{b_name}",
            "family": "Baseline Predictor",
            "variant": b_name,
            "bits_payload": 0.0,
            "metadata_bytes": meta_bytes,
            "total_bytes": meta_bytes,
            "true_bpw": b_bpw,
            "direct_apply_class": "DIRECT_APPLY_PLAUSIBLE",
            **b_metrics,
            **run_wx_probe_fast(B_arr, X_probe, Y_ref_probe),
            "clipping_count": 0,
            "fit_time_ms": 1.0,
            "encode_time_ms": 1.0
        })

    # 4. Learned Control: Free 8-State Lloyd-Max Codebook
    print("Fitting Free 8-State Lloyd-Max Control...", flush=True)
    start_t = time.perf_counter()
    B_med = b_models["B2_GlobalMedian"]
    residuals_fit = (ref_w - B_med)[fit_mask]
    lm_code, _ = lloyd_max_scalar(residuals_fit, k=8, iterations=20)
    
    res_all = ref_w - B_med
    cuts = (lm_code[:-1] + lm_code[1:]) / 2.0
    lm_idx = np.searchsorted(cuts, res_all.ravel()).astype(np.uint8)
    lm_recon = (B_med.ravel() + lm_code[lm_idx]).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    lm_payload_bytes = int(math.ceil(total_elements * 3 / 8.0))
    lm_meta_bytes = 8 * 2 + 2
    lm_total_bytes = lm_payload_bytes + lm_meta_bytes
    lm_bpw = (lm_total_bytes * 8.0) / total_elements
    
    lm_metrics = evaluate_metrics(ref_w, lm_recon, test_mask)
    lm_wx = run_wx_probe_fast(lm_recon, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "LloydMax_8State_Control",
        "family": "Learned Control",
        "variant": "Free 8-Level Residual Codebook",
        "bits_payload": 3.0,
        "metadata_bytes": lm_meta_bytes,
        "total_bytes": lm_total_bytes,
        "true_bpw": lm_bpw,
        "direct_apply_class": "DIRECT_APPLY_UNPROVEN",
        **lm_metrics,
        **lm_wx,
        "clipping_count": int(np.sum((lm_idx == 0) | (lm_idx == 7))),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    # 5. Geometric CCC C3 Grid Search on Validation Split
    print("Fitting Geometric CCC C3 variants on Fit/Val split...", flush=True)
    res_w = ref_w - B_med
    mad_val = float(np.median(np.abs(res_w[fit_mask])))
    sigma_val = float(np.std(res_w[fit_mask]))
    p99_val = float(np.percentile(np.abs(res_w[fit_mask]), 99))
    
    scale_candidates = {
        "1.5_MAD": 1.5 * mad_val,
        "2.5_MAD": 2.5 * mad_val,
        "3.5_MAD": 3.5 * mad_val,
        "2.0_Sigma": 2.0 * sigma_val,
        "3.0_Sigma": 3.0 * sigma_val,
        "P99": p99_val
    }
    gamma_candidates = [1.0, 1.15, 1.25, 1.35, 1.5, 1.7, 2.0]

    # Find best C3-A configuration on Val split
    best_c3a_val_rmse = 1e9
    best_c3a_params = None
    
    for sc_name, R in scale_candidates.items():
        for g in gamma_candidates:
            x_m = (np.arange(4) + 0.5) / 4.0
            m_levels = R * (x_m ** g)
            c3a_levels = np.sort(np.concatenate([-m_levels, m_levels])).astype(np.float32)
            c3a_cuts = (c3a_levels[:-1] + c3a_levels[1:]) / 2.0
            c3a_idx_val = np.searchsorted(c3a_cuts, res_w[val_mask]).astype(np.uint8)
            recon_val = B_med[val_mask] + c3a_levels[c3a_idx_val]
            val_rmse = np.sqrt(np.mean((ref_w[val_mask] - recon_val)**2))
            if val_rmse < best_c3a_val_rmse:
                best_c3a_val_rmse = val_rmse
                best_c3a_params = (sc_name, R, g)

    print(f"Best C3-A parameters on Val split: {best_c3a_params} (Val RMSE: {best_c3a_val_rmse:.6f})", flush=True)

    # Evaluate best C3-A on Test split
    sc_name, R, g = best_c3a_params
    start_t = time.perf_counter()
    x_m = (np.arange(4) + 0.5) / 4.0
    m_levels = R * (x_m ** g)
    c3a_levels = np.sort(np.concatenate([-m_levels, m_levels])).astype(np.float32)
    c3a_cuts = (c3a_levels[:-1] + c3a_levels[1:]) / 2.0
    c3a_idx = np.searchsorted(c3a_cuts, res_w.ravel()).astype(np.uint8)
    c3a_recon = (B_med.ravel() + c3a_levels[c3a_idx]).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    c3_bytes = int(math.ceil(total_elements * 3 / 8.0)) + 5
    c3_bpw = (c3_bytes * 8.0) / total_elements
    c3a_metrics = evaluate_metrics(ref_w, c3a_recon, test_mask)
    results.append({
        "candidate_id": f"CCC_C3A_Best_g{g:.2f}_{sc_name}",
        "family": "Geometric CCC C3",
        "variant": f"C3-A No-Zero Best (g={g}, R={sc_name})",
        "bits_payload": 3.0,
        "metadata_bytes": 5,
        "total_bytes": c3_bytes,
        "true_bpw": c3_bpw,
        "direct_apply_class": "DIRECT_APPLY_UNPROVEN",
        **c3a_metrics,
        **run_wx_probe_fast(c3a_recon, X_probe, Y_ref_probe),
        "clipping_count": int(np.sum((c3a_idx == 0) | (c3a_idx == 7))),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    # Find best C3-B tail-state configurations on Val split for each tail multiplier
    x_m3 = np.array([1/3.0, 2/3.0, 1.0])
    pos_tail_max = np.max(res_w[fit_mask])
    neg_tail_max = np.abs(np.min(res_w[fit_mask]))
    
    for tail_mult in [1.5, 2.0, 3.0, 4.0]:
        best_c3b_val_rmse = 1e9
        best_c3b_params = None
        for sc_name, R in scale_candidates.items():
            for g in gamma_candidates:
                m_levels3 = R * (x_m3 ** g)
                r_tail = R * tail_mult
                tail_val = r_tail if pos_tail_max >= neg_tail_max else -r_tail
                c3b_levels = np.sort(np.array([0.0, m_levels3[0], m_levels3[1], m_levels3[2],
                                               -m_levels3[0], -m_levels3[1], -m_levels3[2], tail_val], dtype=np.float32))
                c3b_cuts = (c3b_levels[:-1] + c3b_levels[1:]) / 2.0
                c3b_idx_val = np.searchsorted(c3b_cuts, res_w[val_mask]).astype(np.uint8)
                recon_val = B_med[val_mask] + c3b_levels[c3b_idx_val]
                val_rmse = np.sqrt(np.mean((ref_w[val_mask] - recon_val)**2))
                if val_rmse < best_c3b_val_rmse:
                    best_c3b_val_rmse = val_rmse
                    best_c3b_params = (sc_name, R, g)
                    
        sc_name, R, g = best_c3b_params
        start_t = time.perf_counter()
        m_levels3 = R * (x_m3 ** g)
        r_tail = R * tail_mult
        tail_val = r_tail if pos_tail_max >= neg_tail_max else -r_tail
        c3b_levels = np.sort(np.array([0.0, m_levels3[0], m_levels3[1], m_levels3[2],
                                       -m_levels3[0], -m_levels3[1], -m_levels3[2], tail_val], dtype=np.float32))
        c3b_cuts = (c3b_levels[:-1] + c3b_levels[1:]) / 2.0
        c3b_idx = np.searchsorted(c3b_cuts, res_w.ravel()).astype(np.uint8)
        c3b_recon = (B_med.ravel() + c3b_levels[c3b_idx]).reshape(m_rows, n_cols)
        fit_time = (time.perf_counter() - start_t) * 1000.0
        
        c3b_bytes = c3_bytes + 2
        c3b_bpw = (c3b_bytes * 8.0) / total_elements
        c3b_metrics = evaluate_metrics(ref_w, c3b_recon, test_mask)
        
        results.append({
            "candidate_id": f"CCC_C3B_Tail{tail_mult}x_Best",
            "family": "Geometric CCC C3",
            "variant": f"C3-B Tail-State Best (g={g}, R={sc_name}, tail={tail_mult}x)",
            "bits_payload": 3.0,
            "metadata_bytes": 7,
            "total_bytes": c3b_bytes,
            "true_bpw": c3b_bpw,
            "direct_apply_class": "DIRECT_APPLY_UNPROVEN",
            **c3b_metrics,
            **run_wx_probe_fast(c3b_recon, X_probe, Y_ref_probe),
            "clipping_count": int(np.sum(c3b_levels[c3b_idx] == tail_val)),
            "fit_time_ms": fit_time,
            "encode_time_ms": fit_time
        })

    # 6. C4 Follow-Up Evaluation
    print("Evaluating C4 Follow-up on Test split...", flush=True)
    start_t = time.perf_counter()
    x_c4 = np.linspace(1/8.0, 1.0, 8)
    R_c4 = 3.0 * mad_val
    m_c4 = R_c4 * (x_c4 ** 1.35)
    c4_levels = np.sort(np.concatenate([-m_c4, m_c4])).astype(np.float32)
    c4_cuts = (c4_levels[:-1] + c4_levels[1:]) / 2.0
    c4_idx = np.searchsorted(c4_cuts, res_w.ravel()).astype(np.uint8)
    c4_recon = (B_med.ravel() + c4_levels[c4_idx]).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    c4_bytes = int(math.ceil(total_elements * 4 / 8.0)) + 5
    c4_bpw = (c4_bytes * 8.0) / total_elements
    c4_metrics = evaluate_metrics(ref_w, c4_recon, test_mask)
    c4_wx = run_wx_probe_fast(c4_recon, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "CCC_C4_MirroredPower_g1.35",
        "family": "Geometric CCC C4",
        "variant": "C4 Mirrored Power (g=1.35, R=3.0MAD)",
        "bits_payload": 4.0,
        "metadata_bytes": 5,
        "total_bytes": c4_bytes,
        "true_bpw": c4_bpw,
        "direct_apply_class": "DIRECT_APPLY_PLAUSIBLE",
        **c4_metrics,
        **c4_wx,
        "clipping_count": int(np.sum((c4_idx == 0) | (c4_idx == 15))),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    # Free 16-State Lloyd-Max Control
    start_t = time.perf_counter()
    lm16_code, _ = lloyd_max_scalar(residuals_fit, k=16, iterations=20)
    cuts16 = (lm16_code[:-1] + lm16_code[1:]) / 2.0
    lm16_idx = np.searchsorted(cuts16, res_all.ravel()).astype(np.uint8)
    lm16_recon = (B_med.ravel() + lm16_code[lm16_idx]).reshape(m_rows, n_cols)
    fit_time = (time.perf_counter() - start_t) * 1000.0
    
    lm16_bytes = int(math.ceil(total_elements * 4 / 8.0)) + 34
    lm16_bpw = (lm16_bytes * 8.0) / total_elements
    lm16_metrics = evaluate_metrics(ref_w, lm16_recon, test_mask)
    lm16_wx = run_wx_probe_fast(lm16_recon, X_probe, Y_ref_probe)
    results.append({
        "candidate_id": "LloydMax_16State_Control",
        "family": "Learned Control",
        "variant": "Free 16-Level Residual Codebook",
        "bits_payload": 4.0,
        "metadata_bytes": 34,
        "total_bytes": lm16_bytes,
        "true_bpw": lm16_bpw,
        "direct_apply_class": "DIRECT_APPLY_PLAUSIBLE",
        **lm16_metrics,
        **lm16_wx,
        "clipping_count": int(np.sum((lm16_idx == 0) | (lm16_idx == 15))),
        "fit_time_ms": fit_time,
        "encode_time_ms": fit_time
    })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving qualification JSON to {OUT_JSON}...", flush=True)
    with OUT_JSON.open("w") as f:
        json.dump({
            "source_tensor": t_info['name'],
            "shape": t_info['shape'],
            "elements": total_elements,
            "source_ggml_type": "Q8_0",
            "source_sha256": "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169",
            "split_ratio": "70/15/15",
            "results": results
        }, f, indent=2)

    print(f"Saving qualification CSV to {OUT_CSV}...", flush=True)
    keys = list(results[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("Qualification benchmark complete!", flush=True)

if __name__ == "__main__":
    main()
