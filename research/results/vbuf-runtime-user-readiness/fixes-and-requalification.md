# DeepSeek numerical fixes and memory requalification

Date: 2026-09-14. Follow-up to the [failed baseline](README.md).

Subsequent work: [bounded parallel expert execution](parallel-execution.md)
records the next performance improvement; timings here remain the earlier
corrected, detailed-trace baseline.

**Result: PASS for the requested same-model correctness and enforced 2 GiB /
two 16-token gate.** Interactive throughput and general release readiness
remain separate: the normal server still takes roughly five minutes per
16-token completion in this configuration.

## Numerical localization

Independent `oracle_capture.cpp` uses the public llama.cpp context evaluation
callback on the original GGUF. `native_capture.cpp` invokes the canonical
vBuf-owned embedding, layer sequence, materialization/residency, and output head
on validated vBuf payload ranges. Both process the same six prompt IDs, then
feed back their own greedy outputs. The oracle is not linked into the runtime.

The earliest boundaries, before correction, were:

| Boundary | Measured discrepancy |
|---|---|
| Position 0, block 0 output | exact agreement |
| Position 0, block 1 attention / computed residual input | exact agreement |
| Position 0, block 1 FFN normalization | max abs `2.5883564949` |
| Position 1, block 0 unrotated Q and V | exact agreement |
| Position 1, block 0 rotated Q | max abs `13.766324997` |

Three semantic defects were corrected:

1. **Missing MoE residual input.** Serial and batched paths added the attention
   residual only for the dense FFN. Every FFN now consumes
   `layer_input + attention_output`, including its residual merge and router.
2. **Wrong rotary/scaling profile.** The existing model-specific direct graph
   used unscaled NEOX half-pair RoPE and plain `1/sqrt(192)` attention scaling.
   The pinned DeepSeek source uses interleaved pairs, YaRN factor 40, original
   context 4096, beta 32/1, and exported log multiplier 0.0707. The independently
   captured attention scale is `0.114721373`. These semantics are explicit in
   `deepseek_v2_lite_semantics.h`; no persistent-format fields were changed.
3. **Incorrect TopK renormalization.** `expert_weights_norm=false` requires
   selecting the full-router softmax probabilities. Re-normalizing the six
   chosen weights inflated the first block-1 weight from `0.135739282` to
   `0.336612582`. Selection/rank order is preserved; expert contributions are
   still accumulated in original TopK order.

Tiny numerical differences remaining after those corrections were amplified
by later quantized matmul input rounding. The direct CPU path now uses GGML
F32 attention and router reductions, with zero-padded/masked 256-position
attention geometry. Padding is transient backend representation, not live KV
positions or a persistent layout change. Explicit fused multiply-add in rotary
interpolation/rotation preserves the pinned CPU rounding path. Without the
interpolation FMA, bitwise agreement first failed at position 9 and then
amplified; the failed control is preserved in `numerical-unfused-control.jsonl`.

### Recorded numerical result

**PASS: all 102,400 logits at all 21 positions are bit-identical**, including
greedy argmax, across the six-token prefill and 16-token completion (the first
generated token comes from the last prefill position). This is 2,150,400 checked
logit values. See `numerical-generation16-fixed.jsonl`, whose final record is:

```json
{"logits_gate": "PASS", "positions": 21, "max_abs_allowed": 0.0}
```

Generated IDs:

```text
8913,11,285,254,6077,280,7239,317,8913,13,185,2640,317,254,6077,280
```

Generated text (leading space preserved):

```text
 Paris, and the capital of France is Paris.
What is the capital of
```

The numerical capture uses a **local source**, diagnostic boundary capture,
and a prepared x86_64 CPU build. It is not a throughput benchmark, not a GPU or
Android result, and not universal bitwise portability. The subsequent strict
gate uses the normal HTTP server and compares streamed text to this separately
recorded oracle output. It executes no reference work in normal inference.

## Anonymous memory attribution

A server-only glibc `mallinfo2` preload sampled heap state. A matched diagnostic
control additionally called `malloc_trim(0)` every 500 ms. Both used the
numerically corrected model, 27 layers, 256 MiB residency, one four-token
request, 4 GiB group limit, and the same source/provider path.

| Metric | Original allocation | Diagnostic trimming |
|---|---:|---:|
| Sampled runtime RSS peak | 2,327,834,624 B | 516,878,336 B |
| Peak malloc arena address space | 2,191,052,800 B | 2,235,707,392 B |
| Peak free bytes inside arenas | 1,884,828,592 B | 1,920,046,336 B |
| Peak live malloc allocation, simultaneous arena-used + malloc-mmap | 471,678,032 B | 471,673,600 B |
| Output | ` Paris, and the` | ` Paris, and the` |
| Source requests / bytes | 6,439 / 6,412,541,856 | 6,439 / 6,412,541,856 |

This attributes the dominant excess to **resident free malloc-arena pages**,
not live tensor ownership: trimming drops RSS while virtually reserved arena
space and peak live allocation remain similar. Counters have distinct scopes;
their independent peaks are not additive. The preload and periodic trim are
observer interventions, so these timings are not production performance claims.
Evidence: `heap-baseline.json`, `heap-trim-control.json`, and adjacent heap and
memory time series.

The production materializer now allocates payload spans of at least 64 KiB with
anonymous page-backed mappings. Their final shared lease calls `munmap`; small
allocations retain the existing aligned allocation path. This prevents large
cross-thread payload allocations/frees from stranding evicted bytes in malloc
arenas. There is no process-global allocator tuning or periodic trimming in
production. The mapping is materialized RAM, not a claim of zero-copy HTTP.

## Strict gate

**PASS.** Two complete, sequential 16-token requests in the same normal server
process, all 27 layers, matched the independent oracle text exactly. The
256 MiB weight-cache setting and swap-disabled 2 GiB cgroup limit were unchanged.
The group included the runtime, loopback payload provider, client, and sampler.
The final run had no allocation preload, trimming control, or oracle execution.

| Metric | First request | Reused session |
|---|---:|---:|
| Generated tokens | 16 | 16 |
| Completed layers / positions | 27 / 21 | 27 / 21 |
| Oracle text match | true | true |
| First-token latency | 44.528 s | 43.899 s |
| Request wall time | 292.447 s | 310.333 s |
| Decode throughput, excluding first token | 0.06051 tokens/s | 0.05631 tokens/s |
| Source requests | 14,906 | 14,673 |
| Requested source bytes | 15,707,595,552 | 15,327,392,544 |
| Materialized bytes, inclusive repeated work | 13,149,179,904 | 12,818,356,224 |
| Peak payload residency counter | 268,379,136 B | 268,401,664 B |
| Active leases / inflight bytes after | 0 / 0 | 0 / 0 |

Whole-run memory and lifecycle:

- Enforced `memory.max`: **2,147,483,648 bytes**; `memory.swap.max=0`.
- Sampled runtime RSS peak: **585,293,824 bytes (558.18 MiB)**.
- Sampled runtime anonymous peak: **572,768,256 bytes (546.23 MiB)**.
- Sampled provider/client RSS peak: **189,652,992 bytes (180.87 MiB)**.
- Kernel cgroup high-water: **2,148,057,088 bytes**, 560 KiB above the limit
  as a transient kernel accounting/charge overshoot. Sampled group maximum
  was exactly 2,147,483,648 bytes. This is an enforced-limit completion, not a
  claim that every instantaneous kernel counter was strictly below 2 GiB.
- Memory-limit pressure/reclaim events: **26,232**; OOM / OOM-kill / group-OOM:
  **0 / 0 / 0**. File cache was reclaimed under the limit.
- Swap peak: **0**. Server exit: **0**, clean shutdown.
- One payload connection across both requests; 29,579 source requests,
  31,034,988,096 requested bytes, and 3,634,105,888 unique requested-range bytes.
  Unique requested bytes are not automatically unique useful tensor bytes;
  no exact overfetch value is inferred from these counters.

The artifact is 5,639,819,878 bytes (about 5.25 GiB), larger than the enforced
group limit. Storage was local NVMe exposed through the existing loopback range
provider. Cache-dropping was advisory, and request 2 reused process/source state;
physical cold-cache, real internet transport, a physical 2 GiB machine, long
contexts, and long-running service stability are not qualified here. The
numeric local-source logit capture and HTTP bounded-memory text gate are
separate recorded qualifications.

Evidence: `deepseek-fixed-2g-256m.json` and its full `.memory.jsonl` time series.
The original failed 2 GiB run remains in `deepseek-2g-256m.json`; the original
incorrect-output 4 GiB control also remains intact. No matched throughput
speedup is claimed against those different-output/different-length workloads.

**Readiness decision:** the requested numerical and excess-memory blockers are
fixed for this pinned model/profile. TPA packaging can build on this qualified
boundary, but the measured 44-second first-token delay and approximately
0.06-token/s decode are still unsuitable for a comfortable interactive release.

## Regression coverage and verification

- `vbuf_deepseek_semantics_contract`: backend-comparison tests for nonzero
  interleaved YaRN, attention magnitude, non-renormalized TopK probabilities,
  and a complete synthetic dense + MoE identity stack through the actual
  serial and batched executors. Checks throw on failure even in Release builds.
- `vbuf_materializer_page_lifetime_contract`: payload remains readable through
  an outstanding lease after release, then `mincore` confirms the mapping is
  gone when the final lease ends. This checks actual OS release, not only a
  logical byte counter.
- Native CTest: 26/26 PASS.
- Existing materializer, residency, tensor-adapter, source-fallback, and runtime
  state tests were also compiled separately with `-O2 -UNDEBUG` and executed:
  5/5 PASS. This avoids relying only on Release builds that disable `assert`.
- Rust workspace tests: PASS.
- Portable graph neutrality: both leakage counters zero.

This work fixes the existing DeepSeek-V2-Lite execution profile. Generic model
discovery/configuration and other models' rotary/routing profiles are not
qualified by these constants. Historical Android and earlier PoC measurements
remain evidence of their original code and scopes.

## Reproduce captures and gate

Use the model preparation/native build commands in the baseline report first.
Then, from the repository root:

```sh
bash research/results/vbuf-runtime-user-readiness/build_capture_tools.sh \
  ${RUN_ROOT}/deepseek-qualification-build ${LLAMA_CPP} ${RUN_ROOT}
${RUN_ROOT}/oracle-capture research-models/DeepSeek-V2-Lite.IQ2_XXS.gguf \
  ${RUN_ROOT}/oracle-generation16 16
${RUN_ROOT}/native-capture research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf ${RUN_ROOT}/native-generation16-final 27 16
python3 research/results/vbuf-runtime-user-readiness/compare_captures.py \
  ${RUN_ROOT}/oracle-generation16 ${RUN_ROOT}/native-generation16-final \
  --logits-max-abs 0 --expected-logits 21

systemd-run --user --unit=vbuf-footprint-fixed-2g \
  --property=MemoryMax=2G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=1800 --property=WorkingDirectory="$PWD" \
  python3 "$PWD/scripts/qualify_runtime_footprint.py" \
  --server ${RUN_ROOT}/deepseek-qualification-build/vbuf_compat_server \
  --semantic-model "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf" \
  --payload "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf" \
  --output ${RUN_ROOT}/footprint-fixed-2g --blocks 27 \
  --capacity 268435456 --tokens 16 --requests 2 \
  --expected-output ${RUN_ROOT}/oracle-generation16/generation.txt
```

Wait for the service to stop before summarizing or starting other compute runs.
Use fresh capture/output directories for new evidence. The final normal gate
has neither `--server-preload` nor `--trim-control` enabled.
