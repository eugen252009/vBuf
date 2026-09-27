# DeepSeek runtime user-readiness qualification

Date: 2026-09-14

**Evidence packaging:** Local absolute paths in captured records are normalized
as `${VBUF_ROOT}`, `${LLAMA_CPP}`, `${RUN_ROOT}`, and qualification-artifact
variables. This removes workstation usernames and temporary-directory names;
measured values, event ordering, model hashes, and result text are unchanged.
Set these variables to local paths when reproducing commands.

**Latest runtime work:** [bounded parallel expert execution](parallel-execution.md)
adds qualified four-worker execution and removes normal-mode history-copying
overhead. The original failed measurements below remain preserved.

**Follow-up:** the numerical and memory defects below have been fixed and the
two-request 2 GiB / 16-token gate now passes. See
[fixes and requalification](fixes-and-requalification.md) for the measured
result and remaining latency limitations. This document preserves the initial
failed baseline.

**Result: NOT USER-READY.** The strict 2 GiB footprint gate failed with a cgroup
OOM. A 4 GiB control completed two short requests but generated a reproducibly
incorrect-looking continuation that disagrees with the same-model behavioral
oracle despite identical input token IDs. Numerical divergence localization is
still required; this is not an exact-logit parity investigation or a fix.

## Scope and provenance

- vBuf starting commit: `73d38721841ec7ff0221bb4350c4c0d81f7af496`.
- Runtime: canonical PoC22-derived `VbufGenerationSession`, exposed by
  `vbuf_compat_server`; `NormalInference`, CPU, existing serial prefill/decode.
- Native GGML pin: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Host: Linux `6.12.107+deb13-amd64`, glibc 2.41, Ryzen 7 5800X, 16 logical
  CPUs, approximately 62 GiB RAM. Tensor-wave CPU execution uses GGML's default
  four threads. The original draft incorrectly treated the opt-in two-thread
  diagnostic override as the default; no such override was enabled in these runs.
- Model repository: `legraphista/DeepSeek-V2-Lite-IMat-GGUF`.
- Model revision: `3048fc1df365e992c92a055324e8fd872e5763b9`.
- GGUF: `DeepSeek-V2-Lite.IQ2_XXS.gguf`, 5,640,619,552 bytes,
  SHA-256 `3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c`.
- Converted vBuf: 5,639,819,878 bytes,
  SHA-256 `2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47`.
- Semantic bootstrap: 3,206,405 bytes with the locator used below. Its byte
  size can differ from historical evidence because the locator differs.
- Manifest: 377 tensors; **27 layers, blocks 0 through 26**. The older
  Step 31R report's 26-block measurement remains valid for its measured scope,
  but its "full-model" label is not the complete layer count of this artifact.
- No production runtime, format, residency, or backend behavior was changed.
  Additions are the research driver, tokenizer probe, and measured evidence.

## Memory and transport boundaries

`systemd-run --user` imposed `MemoryMax=2G` or `4G` and `MemorySwapMax=0`.
Each group included the runtime server, the existing Python loopback HTTP range
provider, and the Python client/sampler. Payloads resided on the local NVMe
filesystem, not `/tmp` (which is tmpfs on this host).

The driver requested `POSIX_FADV_DONTNEED` for this payload before each process
run. This is advisory: physical cold-cache state is **not proven**. Request 2
reused the same runtime/residency/source session and storage cache. No global
cache flush was performed. File-page charges shared with other cgroups and
reclaim behavior mean this is a constrained-process-group qualification, not a
measurement on a physical 2 GiB machine.

Memory sampling was every 250 ms. Per-process RSS/anonymous peaks are sampled;
cgroup `memory.peak` is a kernel high-water mark. Peaks from different scopes
must not be added. The provider/client is reported separately from the runtime.
Source request counts/bytes are instrumented around the existing range handler;
its payload reading/serving behavior is unchanged. No internet transport or GPU
execution was involved in inference. Oracle work ran separately.

## Results

Prompt: `The capital of France is` (raw completion, no chat template).
Input IDs: `[100000, 549, 6077, 280, 7239, 317]`, including BOS.
Weight residency capacity: 268,435,456 bytes (256 MiB) in both runs.

| Gate | 2 GiB group | 4 GiB control |
|---|---:|---:|
| Requested workload | 2 requests × 16 tokens | 2 requests × 4 tokens |
| Completed requests | 0 | 2 |
| Observed output | `PECPECPEC`, then OOM | `PECPECPECROS` on both requests |
| Kernel group peak | 2,147,864,576 bytes | 3,790,340,096 bytes |
| Sampled runtime RSS peak | 1,897,422,848 bytes | 2,028,195,840 bytes |
| Sampled runtime anonymous peak | 1,884,921,856 bytes | 2,015,510,528 bytes |
| Sampled provider/client RSS peak | 185,626,624 bytes | 186,368,000 bytes |
| Swap | disabled; sampled runtime swap 0 | kernel swap peak 0 |
| Result | `oom-kill` | clean server exit; OOM events 0 |

The kernel peak may transiently exceed the configured limit by a small amount;
the 2 GiB run is a failure, not a successful within-budget execution.
The workloads differ: the 4 GiB control is **not** a successful 16-token gate
or a matched memory-budget speed comparison.

The 2 GiB stream delivered its first token at 46.953 s and subsequent tokens at
59.171 and 72.073 s. The final memory sample was at 86.423 s; systemd then
reported a cgroup OOM and terminated the service. No successful request-end
residency or cleanup telemetry exists for this run.

### Completed control requests

| Metric | Request 1 | Request 2, session reused |
|---|---:|---:|
| First-token latency, client-observed | 44.397 s | 42.274 s |
| Four-token request wall time | 79.945 s | 76.925 s |
| Decode tokens/s, after first token | 0.08444 | 0.08662 |
| Source requests | 6,508 | 6,408 |
| Source bytes requested | 6,525,788,064 | 6,336,546,720 |
| Materialized bytes, runtime counter | 5,767,219,200 | 5,614,749,696 |
| Peak payload residency, runtime counter | 268,274,688 | 268,342,272 |
| Evictions | 6,439 | 6,405 |
| Reacquisitions | 5,195 | 6,408 |
| Completed layers / positions | 27 / 9 | 27 / 9 |
| Active leases / inflight bytes after | 0 / 0 | 0 / 0 |

The source reused **one connection** across 12,916 requests. Requested bytes
totaled 12,862,334,784; their unique range union was 1,433,026,144 bytes. The
unique requested union is not necessarily useful tensor payload. Materialized
bytes count repeated work, not unique payload; no exact overfetch figure is
claimed. Source traffic exceeding the artifact size reflects reacquisition.

Decode rate is `(token events - 1) / (last arrival - first arrival)`, so the
first output token is accounted for in prefill/first-token latency. Timers are
inclusive; nested acquisition/compute/copy counters are not summed.

## Correctness gate

The same GGUF, run separately with the local clean llama.cpp checkout
`a97123e497968f3440264c0464a7adc7c999c027`, CPU-only, greedy sampling, serial
prompt processing, and F32 KV, produced:

```text
 Paris, and the capital of France is Paris.
What is the capital of
```

The exact server tokenizer was exercised by `tokenizer_probe.cpp`; both it and
the oracle produced the six IDs above. The server request logs report matching
prompt hash `ba0932473d80e795`.

This is a **failed behavioral agreement gate**, not an attribution of the
numerical defect. The oracle uses a different GGML build and is not the earlier
historical oracle pin. Output disagreement begins at the first token; ordinary
sampling randomness is excluded by greedy selection. Whole-stack logits and
first divergent layer/operation remain unresolved. The oracle uses its own
loader solely for reference behavior; it does not replace vBuf-ML ownership.

See [oracle command and output](oracle.txt). Oracle timings are deliberately
not used as a matched speedup comparison against the limited, loopback-sourced
vBuf runtime.

The tokenizer probe can be rebuilt against the same native artifacts:

```sh
c++ -O2 -std=c++17 \
  research/results/vbuf-runtime-user-readiness/tokenizer_probe.cpp \
  -I integrations/ggml/include \
  -I ${RUN_ROOT}/deepseek-qualification-build/_deps/ggml_source-src/include \
  ${RUN_ROOT}/deepseek-qualification-build/libvbuf_region_executor.a \
  -L ${RUN_ROOT}/deepseek-qualification-build/ggml/src \
  -lggml -lggml-cpu -lggml-base -L rust/target/release -lvbuf_ml -pthread \
  -Wl,-rpath,${RUN_ROOT}/deepseek-qualification-build/ggml/src \
  -Wl,-rpath,"$PWD/rust/target/release" \
  -o ${RUN_ROOT}/deepseek-tokenizer-probe
${RUN_ROOT}/deepseek-tokenizer-probe \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  "The capital of France is"
```

## Readiness decision and next gate

| Requirement | Outcome |
|---|---|
| Pinned source and deterministic conversion | PASS, expected hashes matched |
| Native contracts | PASS, 24/24 |
| Complete model traversal | PASS for two short 27-layer control requests |
| Payload residency counter within 256 MiB | PASS in completed control |
| Total 2 GiB footprint for 16-token generation | FAIL, OOM before completion |
| Same-input behavior agrees with oracle | FAIL |
| Interactive speed | NOT QUALIFIED; measured control is about 12 s/decode token |
| Long context / sustained generation / cancellation recovery | NOT QUALIFIED here |
| APT-ready runtime release | BLOCKED |

The observed RSS growth is predominantly anonymous runtime memory, not merely
the payload's mmap/file cache. It does not by itself prove a leak: live retained
allocations, diagnostic traces, and allocator retention require attribution.
Zero reported leases/inflight bytes after short requests is insufficient to
establish a small process footprint or long-run stability.

**Next gate:** localize and repair same-token full-model numerical divergence,
then attribute runtime anonymous memory and repeat the unchanged strict
2 GiB/16-token gate. Do not optimize or publish the incorrect-output path as a
qualified inference runtime. A broad residency sweep was deferred once these
blocking correctness and total-memory failures were recorded.

## Reproduction

From the vBuf repository root, with the pinned GGUF in `research-models/`:

```sh
cargo build --release --manifest-path rust/Cargo.toml -p vbuf-ml
python3 scripts/build_step18_manifest.py \
  --source research-models/DeepSeek-V2-Lite.IQ2_XXS.gguf \
  --output-dir ${RUN_ROOT}/deepseek-preparation \
  --manifest ${RUN_ROOT}/deepseek-preparation/manifest.json
python3 scripts/convert_gguf_to_vbuf_ml.py \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.gguf \
  ${RUN_ROOT}/deepseek-preparation/manifest.json \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf \
  --integrity none --evidence-dir ${RUN_ROOT}/deepseek-preparation
rust/target/release/vbuf-ml-semantic-bootstrap \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf \
  research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  http://127.0.0.1:18124/model.vbuf
cmake -S integrations/ggml -B ${RUN_ROOT}/deepseek-qualification-build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DVBUF_ML_LIBRARY="$PWD/rust/target/release/libvbuf_ml.so"
cmake --build ${RUN_ROOT}/deepseek-qualification-build -j 8
ctest --test-dir ${RUN_ROOT}/deepseek-qualification-build --output-on-failure

systemd-run --user --unit=vbuf-footprint-256 \
  --property=MemoryMax=2G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=1800 --property=WorkingDirectory="$PWD" \
  python3 "$PWD/scripts/qualify_runtime_footprint.py" \
  --server ${RUN_ROOT}/deepseek-qualification-build/vbuf_compat_server \
  --semantic-model "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf" \
  --payload "$PWD/research-models/DeepSeek-V2-Lite.IQ2_XXS.vbuf" \
  --output ${RUN_ROOT}/footprint-256 --blocks 27 \
  --capacity 268435456 --tokens 16 --requests 2
```

The service runs asynchronously. Inspect its journal and wait until it has
stopped before summarizing or starting another run. Use a new output directory
and unit name for each run; the driver refuses to overwrite prior evidence.
For the control, use `MemoryMax=4G`, `--tokens 4`, unit
`vbuf-footprint-4g-control`, and output `${RUN_ROOT}/footprint-4g-control`.

```sh
python3 scripts/qualify_runtime_footprint.py summarize \
  --run ${RUN_ROOT}/footprint-256 --unit vbuf-footprint-256.service \
  --output research/results/vbuf-runtime-user-readiness/deepseek-2g-256m.json
```

The successful transient systemd unit was already unloaded when queried after
the control run. Its post-run `systemctl show` defaults (`MemoryMax=infinity`,
`MemoryPeak=[not set]`) are not its run configuration: the in-group result
records the actual `memory.max=4294967296`, `memory.swap.max=0`, and kernel peak
before exit. The OOM run's retained failed unit records its limits and peak.

Raw streams, server logs, systemd evidence, and compact summaries are in
`deepseek-2g-256m.json` and `deepseek-4g-256m-control.json`. Complete 250 ms memory
samples are preserved in the adjacent `.memory.jsonl` files. Model artifacts
and native build products remain untracked.
