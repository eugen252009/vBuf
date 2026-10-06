# Qwen3-14B 26/14 two-GPU capacity ladder qualification

Status: **the explicit 26/14 experimental path completed the sequential 2K, 4K, 8K, 16K, and 32K capacity gates, including a natural-language 32,768-position prefill plus 32 incremental decode steps.** This is an experimental workload qualification, not production qualification. The canonical single-GPU path remains the default, production capacity remains **1,032**, and no 32K/25-15 production placement was enabled.

## Model, devices, and input identity

- Semantic model identity: Qwen3-14B Q4_K_M, 40 blocks, 443 tensors, vBuf semantic SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Device 0: RTX 3060, embedding and blocks 0–25; 287 tensors / `5,561,288,704` payload bytes. Device 1: RTX 2080 SUPER, blocks 26–39 and final norm/head; 156 tensors / `3,434,505,216` payload bytes. The counts and payload sums reconcile exactly; every tensor is resident once.
- The F32 hidden boundary is transferred through per-session CUDA-pinned host memory. No P2P/NVLink is assumed. The 26/14 assignment and runtime default were not changed.
- The large input is a repeated natural-language sentence (a controlled, repetitive test—not a realistic conversation or quality evaluation). The source text SHA-256 is `4b2cb07876addf90c102ebf74f7e15b837d8bd4965b23fd4380050fd47b98794`; the 32,768-token CSV SHA-256 is `23891eeb02382f8c8d517ccd0d039bbd3e3c72f6ef428ed13946798279a6ad44`. It has no added BOS. Both input files and the tokenizer log are in `raw/`.
- The 64-capacity baseline was rerun first with synthetic token IDs `0..31`, prefixes 1, 2, 4, 8, 16, and 32. All greedy tokens matched the canonical single-device reference; all six cases passed the *previously declared, test-set-specific* relative-RMS ≤0.02 and cosine ≥0.9998 bounds. These are not universal thresholds. A/B/A isolation, the existing failure points, and fresh-session recovery passed. Raw output is `raw/small-context-final.log` (SHA-256 `c588c75c88fba8c0b7564bae5864fab85326b564eabf3d6609780b7929b34d2f`).

## Sequential capacity and memory gates

The capacity harness checked device ownership and exact KV/packed-V tensor geometry, separate model/KV/prefill/decode buffers, pinned staging, finite sampled F16 K/V, finite hidden/logits, historical K/V preservation, boundary bytes, and the final logical length. Production admission was never raised.

| Capacity | Chunk | Main prefix + append | Minimum free VRAM during that gate (3060 / 2080 SUPER, MiB) | Result |
|---:|---:|---:|---:|---|
| 2,048 | 32 | 512 + 4 | 6,274 / 4,046 | Pass; reset/replay, nontrivial failure and same-session recovery |
| 4,096 | 32 | 1,024 + 4 | 6,026 / 3,747 | Pass |
| 8,192 | 32 | 2,048 + 4 | 5,532 / 3,459 | Pass; reset/replay |
| 16,384 | 32 | 4,096 + 4; 8,192 + 8; 16,352 + 8 | 4,544 / 2,857 (minimum from the 4K-prefix gate) | Pass; 8K reset/replay and near-capacity persistent decode also passed |
| 32,768 | 32 | 32,736 + 32 = **32,768** | 2,568 / 1,780 in final run | Allocation, short smoke, exact-fit full run, overflow rejection, and recovery passed |

A separate 16-row chunk run was measured at 8K context (details below). Its additional graph scratch is reflected in those measurements, not in the 32-row capacity table. The first 2K/32-row capacity attempt also stopped at executor construction because the initial scratch guess was undersized; no inference was advanced to the next capacity until the qualification-only bound was increased and the rung rerun successfully. Later, the first 16-row / 16K-capacity executor construction exposed an undersized scratch bound before inference (`170,811,392` bytes required vs `167,772,160` available). The qualification-only estimate was increased for 16-row graphs and the same test then passed with a 168-MiB buffer. That diagnostic failure is retained as `raw/capacity-16384-prefix8192-chunk16-scratch-fail.log`; neither sizing miss is counted as a passed gate.

### 32K preallocation gate

Before attempting the 32K allocation, the conservative 16K observed execution minima were reduced by the exact known allocation deltas. From 16,384 to 32,768, device 0 adds `1,744,830,464` bytes KV + `33,554,432` packed-V + `251,658,240` prefill scratch = `2,030,043,136` bytes. Device 1 adds `939,524,096` + `33,554,432` + `251,658,240` = `1,224,736,768` bytes. Decode scratch is constant; pinned boundary storage is host memory. Starting at the earlier 16K minima, projected free was `2,734,817,280` bytes (3060) and `1,771,241,472` bytes (2080S). After holding back 512 MiB, the projections were `2,197,946,368` and `1,234,370,560` bytes. This cleared the allocation-only go/no-go screen without relying on a different split. The 32K allocation-only run then measured `2,804,023,296` and `1,851,785,216` free bytes after session/executor construction, close to the known-allocation projection. Longer runs showed additional transient/background variation, recorded below.

### Allocation geometry and ownership

All quantities below are per device except the layer counts. KV is F16, 4,096 bytes per layer per token; packed-V is a separate tensor in the same local KV allocation. The decode/control scratch is 32 MiB/device at every capacity. The final qualified 32-row prefill scratch is separate and grows by the explicit formula shown in source/logs.

| Capacity | Local KV (3060 / 2080S bytes) | Packed-V per device | 32-row prefill scratch per device | Pinned boundary bytes |
|---:|---:|---:|---:|---:|
| 2,048 | 218,103,808 / 117,440,512 | 4,194,304 | 73,400,320 | 655,360 |
| 4,096 | 436,207,616 / 234,881,024 | 8,388,608 | 104,857,600 | 655,360 |
| 8,192 | 872,415,232 / 469,762,048 | 16,777,216 | 167,772,160 | 655,360 |
| 16,384 | 1,744,830,464 / 939,524,096 | 33,554,432 | 293,601,280 | 655,360 |
| 32,768 | 3,489,660,928 / 1,879,048,192 | 67,108,864 | 545,259,520 | 655,360 |

Thus observed KV exactly follows `26 × 4,096 × capacity` and `14 × 4,096 × capacity`. No mirrored layer KV, unexplained KV padding, or second packed-V allocation appeared. The 32-row scratch sizing in qualification mode is `max(64 MiB, 3 × capacity × 32 × 40 × sizeof(float) + 40 MiB)`; the 16-row qualification sizing uses the same score term with a 48-MiB graph overhead, based on the measured graph requirement above. The production/nonexperimental capacity policy and the 32-row default are unchanged.

At 32K, model payload + KV/packed-V + prefill scratch + decode scratch reconciled against backend memory reports. The final run reported 1,975,582,720 free bytes on the 2080S immediately after session/executor allocation and a minimum of 1,866,203,136 bytes during the run. Device 0 had 2,804,023,296 free after allocation and a 2,692,874,240-byte minimum. A separate 32K repeat saw the 2080S minimum fall to 1,495,269,376 bytes while unrelated display/driver usage was higher; both raw runs are retained. This variability matters for the split projection below.

## 32K boundary, KV, admission, and persistent-decode result

The final-source full gate used the exact 32,736-token prefix plus 32 generated tokens, filling capacity exactly. It processed 1,023 prefill chunks of 32 and 32 one-token decode steps. Result: final logical length 32,768; 1,055 staged boundary handoffs / `671,088,640` transferred bytes; 41 selected end-to-end byte audits / `6,553,600` audited bytes. Selected audit endpoints included 8K, 16K, 24K, 32,736, and every generated position through 32,768. Every audit matched the pinned source bytes.

At visible positions 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384, 24,576, and 32,736, the harness sampled K/V for layers 0, 12, 25, 26, 32, and 39; samples were finite and remained byte-identical after the 32 decode steps. New appended K/V rows were finite. Final hidden and logits were finite. Exact-fit admission passed; an additional token was rejected before execution without changing committed position; a valid same-session request then recovered. A separate 128-token-prefix/4-token-append 32K smoke also passed.

The final full-run prefill timer was `616,159,414,509 ns` (`602.3 ms` per 32-row chunk, 53.1 prefix tokens/s); 32 decode steps took `11,954,138,670 ns` (`373.57 ms/token`). The archived planning projection was 595.819 ms per representative 32-row prefill and 375.892 ms/decode. These are close (+1.09% prefill, −0.62% decode), but scopes are not identical: planning sums representative block measurements and omits final-norm work; this gate includes callback/memory probes and selected boundary audits. Treat the comparison as measured-versus-projection evidence, not a clean benchmark or transport ceiling.

The test input repeats one sentence. This proves the tested full-capacity execution/ownership path, not natural-language quality, parity against a single-GPU oracle at 32K, broad prompt coverage, concurrency, multi-hour stress, or production readiness.

## 16-row versus 32-row prefill

Both modes were exercised with the same natural token stream and explicit two-device placement. At capacity 16,384 and prefix 8,192 + 8 decode, both produced the same eight greedy token IDs (`87856,279,20579,6894,323,2750,11,323`). Token fingerprints matched. Final hidden/logit byte fingerprints differed between chunk sizes, so this is token parity for this input, **not bitwise or numerical parity**; no new universal tolerance is asserted.

| Capacity / prefix | Chunk | Prefill time | Prefix throughput | Decode time/token | Handoffs |
|---|---:|---:|---:|---:|---:|
| 8,192 / 4,096 | 32 | 15.663 s | 261.5 tok/s | 61.97 ms | 132 |
| 8,192 / 4,096 | 16 | 22.480 s | 182.2 tok/s | 61.97 ms | 260 |
| 16,384 / 8,192 | 32 | 55.548 s | 147.5 tok/s | 112.45 ms | 264 |
| 16,384 / 8,192 | 16 | 77.123 s | 106.2 tok/s | 110.77 ms | 520 |
| 16,384 / 16,352 | 32 | 110.814 s | 147.6 tok/s | 112.76 ms | 519 |
| 32,768 / 32,736 | 32 | 616.159 s | 53.1 tok/s | 373.57 ms | 1,055 including decode |

These are instrumented capacity-gate timings; all handoffs were audited in the smaller all-audit cases, while long cases audited selected handoffs. The 16-row chunk doubles transfer count for the same prefill rows and was slower in both matched measurements. Chunk 32 remains the default.

## Failure, reset, and recovery

- At capacity 2,048, a 512-token prefix was fully committed. The harness then armed `ExecutionAfterBoundaryD2H` for the next decode step: early-device work wrote layer-25 K/V at position 512, the late path failed, and logical length reset to zero. The physical partial K/V write was intentionally not rolled back. A valid same-session request then succeeded.
- Reset/replay was bitwise stable with allocation reuse at capacities 2,048 and 8,192, and again at capacity 16,384 after an 8,192-token prefix plus eight generated tokens. The 16K replay preserved logits, hidden, tokens, and allocation handles.
- The baseline small-context suite retained its before/after-boundary and late-stage failure tests and fresh-session recovery. At 32K, exact-fit, overflow-before-execution, and recovery all passed.

## 25/15 split projection (no split change or run)

Moving block 25 to the RTX 2080 SUPER would add its measured `210,166,784`-byte tensor payload and one 32K layer of KV (`134,217,728` bytes), total **344,384,512 bytes (328.4 MiB)**, to the limiting device. The packed-V amount is unchanged. Applying that increment to the lower observed 2080S full-capacity free minimum (`1,495,269,376` bytes) leaves about `1,097.6 MiB`; after preserving a 512-MiB stress allowance, only about **585.6 MiB** remains. This is too marginal to recommend as the next test under the observed desktop variability. The 25/15 split remains an untested arithmetic projection; do not change placement or infer it is safe. Retest only after establishing a larger, controlled free-VRAM reserve.

## Implementation and verification

The qualification gate now supports explicit 16- and 32-row chunks for the two-device harness/runtime; default chunk 32 and the single-device executor remain unchanged. Experimental capacity scratch sizing is opt-in and confined to `allow_experimental_capacity`; production admission remains capped at 1,032. The 16-row and 32-row choices alter batch shape/scheduling, not the declared Qwen CUDA equations.

Built the CUDA qualification targets in `/tmp/vbuf-qwen-multigpu-cuda-build`. The following passed after the final source changes:

- Repeated 64-capacity small-context CUDA/reference qualification.
- `vbuf_qwen3_cuda_ownership_contract` and `vbuf_qwen3_model_admission_contract` under CTest.
- Direct `vbuf_qwen3_runtime_admission_contract` run: `qwen3_runtime_metadata_admission=PASS missing_source_fails_closed=YES`.
- 16K reset/replay, 32K exact-fit/overflow/recovery, and the final 32K full-prefix run.
- `git diff --check`.

No reset, clean, or stash operation was performed; the existing worktree content and protected stashes were preserved. The capacity-32,768 full run does **not** change or supersede the production-qualified capacity of 1,032.

## Post-commit baseline revalidation

After baseline commit `598c1f6e177557bdfdcb2a7762cc8cceb373b0e2`, the targeted build had no pending work; `vbuf_qwen3_cuda_ownership_contract` and `vbuf_qwen3_model_admission_contract` passed under CTest, and the runtime-admission executable passed its fail-closed contract. The 64-capacity small-context test passed again. The full 32K run was repeated with the committed source: prefix 32,736 + 32 decode reached 32,768; 41 boundary audits passed; historical/device-local KV checks, exact fit, overflow rejection and recovery passed. Final-run minimum free VRAM was 2,568 MiB (3060) / 1,486 MiB (2080S). Raw outputs are `raw/stage1-committed-baseline-small-context-rerun.log` (SHA-256 `a0a14d46ed97744e9bc2f2d88113acab9a49c9603a5a7790c0a510a7d335a7df`) and `raw/stage1-committed-baseline-32k-rerun.log` (SHA-256 `bac1866728846f6c52c3191ce270525c70b69ecd7083d334b61521348724426b`). This is a repeat of the already committed baseline, not an optimizer change.

## Raw evidence

All run logs, the exact prompt/token input, and tokenizer output are in `raw/`. The final-source 32K full-run log is `raw/capacity-32768-prefix32736-append32-final.log` (SHA-256 `d9cbfe2e61bf2eb8959239f9d047a68ae0099d26c2c3397c7ef817c2fad28f0b`). The earlier full repeat is kept separately to show VRAM variability. The initial 16-row scratch-sizing miss is explicitly labeled diagnostic and is not included as a passed run.
