# Bounded parallel expert execution

Date: 2026-09-14. Continuation of [correctness and memory qualification](fixes-and-requalification.md).

**Result:** selected-expert CPU work runs independently on a reusable bounded
pool, with exact oracle logits and successful 2 GiB generation gates. On this
workload the parallel path reduced two-request completion time by about 28%
relative to the new trace-disabled serial path (1.40x speedup). This is an
end-to-end result including acquisition-wave/cache effects, not a claim of
fourfold compute scaling or universal model support.

## Execution structure

```text
ordered token / layer state
    -> attention and FFN normalization
    -> row-local router and TopK
    -> vBuf-ML prepares a byte-bounded wave of selected tensors
    -> independent expert jobs on persistent workers
         rank 0 -> result[0]
         rank 1 -> result[1]
         ...
    -> join/drain wave; release leases
    -> remaining selected experts, if needed
    -> accumulate result[0], result[1], ... in original TopK rank order
    -> shared expert / residual / next layer
```

The expert-compute portion is an embarrassingly parallel map followed by an
ordered reduction: jobs share only read-only input and ready weight views and
write distinct output slots. Layer dependencies, causal attention/KV updates,
and autoregressive token feedback remain ordered. This implementation does not
introduce simultaneous user generations or position-level parallel execution.

- `BoundedExecutor` owns reusable worker threads and drains every batch before
  returning or propagating its lowest-rank exception. It never owns sources.
- vBuf-ML prepares the selected gate/up/down tensors before compute. The backend
  receives ready validated tensors through `PreparedExpertMaterializer`, which
  cannot perform source I/O or alter residency policy.
- Admission is bounded by **both** worker count and the residency byte budget.
  If even one complete expert cannot be prepared within that budget, the
  existing tensor-at-a-time serial path is used and a fallback is counted.
- A four-worker wave can issue at most twelve selected-weight requests through
  the existing materializer workers. Transfer and hashing can overlap. The
  measured HTTP source still uses one connection; this is not a multi-connection
  transport qualification.
- The controller retains real leases until all expert jobs have synchronized.
  Results are indexed by router rank, never completion order. Source or compute
  failure drains outstanding work before releasing leases and prevents a final
  merge of partial results.
- Qualification mode stays serial with its existing reference work. Normal
  inference executes no oracle work.

## Configuration

The server now exposes:

```text
--expert-workers N   1..6; default 1 disables the expert pool
--expert-threads N   1..8 per parallel expert; default 1
                    workers * threads must be <= 32
--runtime-trace     opt in to detailed diagnostic history in normal inference
--help              describe the startup options
```

The qualified parallel setting is **four expert workers, one compute thread per
expert**. Non-expert TensorWave operations retain GGML's default four-thread
team; the small attention/router reductions use their existing two-thread path.
The compute-thread product limit does not count the separately bounded
materialization workers. Other worker/thread combinations are configurable but
their performance has not been measured here. `--expert-threads` with a
non-default value requires `--expert-workers > 1`.

With the prepared model and the existing local range provider running:

```sh
${RUN_ROOT}/deepseek-qualification-build/vbuf_compat_server \
  --semantic-model research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  --source-url http://127.0.0.1:18124/model.vbuf \
  --blocks 27 --capacity 268435456 --max-new-tokens 16 \
  --expert-workers 4 --expert-threads 1
```

The provider can be started in a separate terminal with:

```sh
python3 scripts/range_server.py \
  --file research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf --port 18124
```

## Remove the tracing bottleneck separately

The normal path repeatedly copied its growing materialization/residency history
inside attention and expert execution. This cost grew with generated length.
Detailed histories are now opt-in for normal private sessions, while cumulative
materialization/reload/eviction counters remain available independently. The
qualification and historical shared-residency diagnostic paths retain tracing.

A same-model, same-source, same-budget four-token control measured:

| Normal mode | Request time | First token | Decode tokens/s | Requests / bytes |
|---|---:|---:|---:|---:|
| Detailed history enabled | 86.304 s | 48.989 s | 0.08043 | 6,439 / 6,412,541,856 |
| Detailed history disabled | 52.158 s | 34.528 s | 0.17018 | 6,439 / 6,412,541,856 |

Both produced ` Paris, and the`. Evidence: `parallel-trace-on.json` and
`parallel-trace-off.json`, with adjacent full memory samples. These are single
controlled runs, not a statistical latency distribution. The tracing ablation
is separate from the parallelism comparison below.

The materialized-byte counter now covers **all** successful materializations,
including embedding and output-head work; older trace-derived counters omitted
some of these. Reload bytes count repeated successful physical source ranges
within the materializer instance. Source-request bytes remain independently
measured by the provider. Do not compare old/new materialized-byte counters as
though they had identical coverage.

## Parallelism comparison

Host/model/pins are those in the linked baseline: Ryzen 7 5800X, Linux x86_64,
native GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, 5,639,819,878-byte
DeepSeek-V2-Lite IQ2_XXS artifact. The prompt is `The capital of France is`, six
tokens including BOS. All 27 layers execute. Each process serves two consecutive
16-token requests. Both variants use a 256 MiB weight-residency cap, enforced
2 GiB group limit, swap disabled, and detailed history disabled.

| Metric | Serial request 1 | Serial request 2 | Parallel request 1 | Parallel request 2 |
|---|---:|---:|---:|---:|
| Completion seconds | 121.469 | 119.501 | 93.934 | 78.627 |
| First-token seconds | 34.186 | 32.800 | 25.022 | 9.824 |
| Decode tokens/s | 0.17186 | 0.17301 | 0.21768 | 0.21802 |
| Source requests | 14,906 | 14,673 | 13,981 | 13,016 |
| Source bytes | 15,707,595,552 | 15,327,392,544 | 15,192,451,872 | 14,496,499,488 |
| Generated tokens | 16 | 16 | 16 | 16 |
| Oracle text match | true | true | true | true |

The two completion times sum to **240.970 s serial vs 172.560 s parallel**.
The derived speedup is approximately **1.40x** and elapsed-time reduction is
approximately **28.4%**. Preparation changes cache access/admission order and
reacquisition, so this includes more than compute concurrency. In particular,
the reused-session first-token improvement is workload/cache-specific.

Each parallel request recorded:

- **3,276 expert jobs**, **1,092 waves**, and **four simultaneously active workers**.
- Peak prepared selected weights: **12,435,456 bytes**, below the 256 MiB budget.
- No serial fallback for this model/budget.
- 27 completed layers and 21 completed positions.
- Zero active leases, lease bytes, and in-flight bytes after completion.

Memory/lifecycle over the paired runs:

| Metric | Serial | Parallel |
|---|---:|---:|
| Sampled runtime RSS peak | 485,801,984 B (463.30 MiB) | 537,501,696 B (512.60 MiB) |
| Kernel group high-water | 2,147,979,264 B | 2,148,139,008 B |
| OOM / OOM-kill | 0 / 0 | 0 / 0 |
| Swap peak | 0 | 0 |
| Server exit | 0 | 0 |
| Source connections across both requests | 1 | 1 |
| Unique requested-range bytes | 3,634,105,888 | 3,634,105,888 |

The parallel kernel high-water counter was 640 KiB above the configured 2 GiB
limit; this is an enforced-limit completion with reclaim, not a claim that
every instantaneous counter stayed below the limit. The group includes the
provider/client and file cache; its peak must not be added to process RSS.
Cache dropping used advisory `POSIX_FADV_DONTNEED`. Physical cold-cache state is
not proven; the second request reuses process/residency/source state. These are
local-NVMe/loopback results, not internet, GPU, Android, or physical-2-GiB-machine
qualification. Unique requested bytes are not independently qualified unique
useful payload; exact overfetch is not separately reported.

Evidence: `parallel-serial16.json`, `parallel-workers4-16.json`, and their memory
time series. A final integration rerun after concurrency error-path and trace
publication hardening is recorded separately in `parallel-final16.json`:

- Both 16-token requests completed with exact oracle text, clean shutdown,
  zero OOM events, zero swap, and zero outstanding leases/in-flight bytes.
- Request times: **95.794 / 80.995 s**; first-token times **25.769 / 10.204 s**.
- Decode rates: **0.21421 / 0.21190 tokens/s**.
- Sampled runtime RSS peak: **529,604,608 bytes (505.07 MiB)**.
- Kernel group high-water: **2,148,085,760 bytes**, 588 KiB above the configured
  limit; the limit remained enforced with reclaim and no OOM.
- Source request/byte totals and selected-expert worker/wave counts matched
  the earlier parallel comparison exactly. The rerun is confirmation, not a
  replacement chosen to improve the comparison's timing result.

## Numerical and concurrency gates

- Four-worker **local-source** capture matched all **2,150,400 logits** exactly
  across 21 positions, feeding back its own generated token IDs. This is
  separate from the HTTP text/memory gate. See `parallel-workers4-parity.jsonl`.
- `parallel_executor_contract` forces out-of-order completion, checks bounded
  overlap, and verifies a rounding-sensitive rank-ordered merge. It checks
  exception draining, deterministic error selection, reuse after failure,
  worker bounds, and overflow rejection.
- `deepseek_semantics_contract` exercises the actual parallel model functions,
  source-failure cleanup/recovery, compute-failure cleanup, low-budget serial
  fallback, and serial qualification mode.
- `runtime_trace_contract` verifies counters with tracing off/on, checked
  in-flight budget arithmetic, and completed source-result snapshots. Source
  results and payload hashes are published under the trace-reader mutex.
- Native CTest: **28/28 PASS**. Rust workspace: **PASS**. Both neutrality counters:
  **zero**.
- The existing materializer, residency, adapter, source-fallback, and runtime
  state contracts also passed separately with `-O2 -UNDEBUG` (**5/5**).
- ThreadSanitizer: **PASS** for the bounded executor contract and for the
  materializer/residency/range-source trace contract. This is scoped sanitizer
  coverage, not a whole-model sanitizer qualification.

## Reproduce the bounded run

After the model/native preparation and independent oracle capture documented in
the previous report:

```sh
systemd-run --user --wait --pipe --unit=vbuf-experts-parallel16 \
  --property=MemoryMax=2G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=900 --property=WorkingDirectory="$PWD" \
  python3 "$PWD/scripts/qualify_runtime_footprint.py" \
  --server ${RUN_ROOT}/deepseek-qualification-build/vbuf_compat_server \
  --semantic-model "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf" \
  --payload "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf" \
  --output ${RUN_ROOT}/experts-parallel16 --blocks 27 \
  --tokens 16 --requests 2 --expert-workers 4 --expert-threads 1 \
  --expected-output ${RUN_ROOT}/oracle-generation16/generation.txt
```

Use a fresh unit/output directory on a rerun. Change `--expert-workers` to `1`
for the serial control. `native-capture` accepts a final worker-count argument:
`native-capture SEMANTIC PAYLOAD OUTPUT 27 16 4`; compare its captures using
`compare_captures.py ORACLE OUTPUT --logits-only --logits-max-abs 0 --expected-logits 21`.

The next scaling opportunities are the remaining serial acquisition boundaries
and further prefill batching. The current end-to-end decoder still measures
about 0.22 tokens/s, so this is a qualified parallel execution step rather than
an interactive-throughput release claim.
