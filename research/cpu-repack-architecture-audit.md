# vBuf / ggml CPU_REPACK Architecture Audit

## Scope and evidence

This audit uses the committed DeepSeek-V2-Lite.IQ1_S Orange Pi RV2 evidence in
`benchmark-results/vbuf-ml-rvv-gguf-comparison/`. That evidence is not changed
by this report. The source references below are from the llama.cpp CPU/loader
sources used for the audit; the benchmark's pinned source identity remains
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`.

The controlled baseline is:

| Measurement | GGUF | vBuf |
|---|---:|---:|
| Warm total median | 103898.894 ms | 77366.161 ms |
| Execution-ready median | 101750.605 ms | 75107.327 ms |
| Payload materialization median | 99749.641 ms | 72626.880 ms |
| CPU_REPACK median | 32549.910 ms | 40991.643 ms |
| Repacked tensors | 28 | 28 |
| Repacked bytes | 2.527 GiB | 2.527 GiB |
| Peak RSS median | 4.780 GiB | 7.282 GiB |

The baseline contains no repack-disabled run. A same-RV2 performance comparison
was not run during this audit: the available remote build has the integration
binary but not the source/build recipe needed to create a separate patched
`GGML_CPU_REPACK=OFF` vBuf binary without touching the benchmark setup. The
correctness conclusion below is therefore source-proven, while the performance
delta without repacking remains an explicit measurement gap.

## A. CPU_REPACK decision chain

The actual chain is:

1. `llama_model_base::load_tensors()` builds the CPU buffer list. With
   `use_extra_bufts`, the CPU backend's extra buffer types are appended before
   the ordinary CPU buffer (`src/llama-model.cpp`, `make_cpu_buft_list()` and
   `load_tensors()`).
2. `ggml_backend_cpu_get_extra_buffer_types()` registers `CPU_REPACK` when
   `GGML_USE_CPU_REPACK` is compiled (`ggml/src/ggml-cpu/ggml-cpu.cpp`).
3. For every model tensor, `llama_model_loader::create_tensor()` maps the model
   tensor role to an operation (`llm_tensor_info_for`, with special handling
   for bias and LoRA suffixes), then calls `select_weight_buft()`.
4. `select_weight_buft()` tests buffer types in order using
   `weight_buft_supported()`. That function constructs a representative
   operation, attaches a temporary zero-size tensor buffer of the candidate
   type, and calls `ggml_backend_dev_supports_op()`.
5. For `CPU_REPACK`, `extra_buffer_type::supports_op()` accepts only:
   - `GGML_OP_MUL_MAT` with a rank-2 weight, or `GGML_OP_MUL_MAT_ID` with a
     rank-3 weight;
   - source 0 in the CPU_REPACK buffer;
   - a source 0 type/shape for which `ggml_repack_get_optimal_repack_type()`
     returns traits;
   - an input source that is host-accessible and `GGML_TYPE_F32`.
6. `ggml_repack_get_optimal_repack_type()` first checks quant type, then
   architecture predicates, then shape predicates. For the RVV cases relevant
   here, the exact predicates are:
   - compile-time `__riscv_zvfh`;
   - runtime `ggml_cpu_has_riscv_v()`;
   - runtime VLEN exactly 256 bits in the switch (`__riscv_vlenb()*8`);
   - `cur->ne[1] % 16 == 0`;
   - type `GGML_TYPE_IQ4_NL` or `GGML_TYPE_Q2_K`.
7. If selected, `ggml_backend_cpu_repack_buffer_init_tensor()` stores the
   traits in `tensor->extra`. The model loader allocates a CPU-backed
   CPU_REPACK buffer and calls `ggml_backend_cpu_repack_buffer_set_tensor()`.
8. `set_tensor()` asserts a whole-tensor transfer, then invokes the traits'
   `repack()` method. The RVV traits are instantiated as
   `tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>` and
   `tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K>`.
9. During execution, the same traits are selected by
   `ggml_cpu_extra_compute_forward()` for `MUL_MAT`/`MUL_MAT_ID`, and dispatch
   to the `*_16x1` GEMV/GEMM kernels.

This is a CPU backend/kernel selection chain. It is not a vBuf validation rule,
not a GGUF semantic rule, and not a hardware requirement.

## B. Tensor classification

The 28 tensors are exactly the tensors for which the RVV predicates above are
true in this model. The benchmark trace is the authoritative per-run list;
the unique list is:

| Tensors | Source -> destination | Shape (GGUF order) | Bytes | Operation/consumer |
|---|---|---|---:|---|
| `blk.0.ffn_down.weight` | IQ4_NL -> IQ4_NL_16x1 | `[10944, 2048]` | 12607488 | `MUL_MAT`, RVV IQ4_NL 16x1 GEMV/GEMM |
| `blk.1` through `blk.26`. `ffn_down_exps.weight` | IQ4_NL -> IQ4_NL_16x1 | `[1408, 2048, 64]` | 103809024 each | `MUL_MAT_ID`, RVV IQ4_NL 16x1 GEMV/GEMM |
| `blk.1.ffn_down_shexp.weight` | Q2_K -> Q2_K_16x1 | `[2816, 2048]` | 1892352 | `MUL_MAT`, RVV Q2_K 16x1 GEMV/GEMM |

The middle row contains 26 tensors, so the count is `1 + 26 + 1 = 28`.
The tensor dimensions and byte sizes come from the matching GGUF descriptor;
the vBuf payload parity check established the same payload sizes. The model has
other `ffn_down_shexp` tensors, but those are IQ1_S and have no RVV 16x1
repack trait. The other 349 tensors fail at least one predicate, most commonly
quant type, operation role, or the absence of a registered RVV repack trait.

The selection is therefore a combination of tensor representation, tensor
shape, operation role, compile-time ISA availability, runtime RVV/VLEN, and
the CPU backend's policy of putting a compatible extra buffer before CPU. It
is not selected by tensor name or by vBuf.

## C. Physical transformation

Both transformations preserve size. They are byte/layout permutations with
the source quantization semantics retained; they do not introduce new numeric
scales or change the quantized values.

### IQ4_NL -> IQ4_NL_16x1

The source `block_iq4_nl` is 18 bytes: one FP16 delta and 16 packed bytes for
32 four-bit values (`QK4_NL=32`). For each output block, the repacker gathers
16 source rows at the same K block. The destination `block_iq4_nlx16` contains:

```text
d[16]       = source-row deltas, unchanged
qs[256]     = source-row packed bytes in row-interleaved order
```

With interleave size 1, destination byte `i` comes from source block
`i % 16`, byte `i / 16`. Thus 16 independent 18-byte blocks become one
296-byte block. `16 * 18 == 296`; alignment is supplied by the buffer type and
the source dimensions require `ne[1] % 16 == 0`. The RVV kernel loads 16
columns per vector tile and performs the nibble lookup in that order.

### Q2_K -> Q2_K_16x1

The source `block_q2_K` is 84 bytes: 16 scale/min bytes, 64 packed 2-bit
quant bytes, and two FP16 super-scales (`QK_K=256`). The destination
`block_q2_Kx16` is 1344 bytes:

```text
d[16]        = source d values, unchanged
dmin[16]     = source dmin values, unchanged
scales[256]  = 16 source scale arrays reordered into four vector groups
qs[1024]     = 16 source 64-byte quant arrays interleaved one-byte at a time
```

For `qs`, output byte `i` reads source column `i % 16` and source byte
`i / 16`. For `scales`, the source sub-block order is explicitly grouped as
even-low `[0,2,4,6]`, odd-low `[1,3,5,7]`, even-high `[8,10,12,14]`, and
odd-high `[9,11,13,15]`, with all 16 columns inside each group. This matches
the RVV phase loop's vector loads. `16 * 84 == 1344`, so source and
destination bytes are equal.

The destination can reconstruct the source losslessly by reversing these
permutations. The transformation is not a new quantization pass.

## D. Correctness requirement

**NO, for correct execution in general.** It is required only by the selected
`*_16x1` optimized kernels.

Evidence:

- The ordinary CPU type table maps `Q2_K` to `ggml_vec_dot_q2_K_q8_K` and
  `IQ4_NL` to `ggml_vec_dot_iq4_nl_q8_0` (`ggml-cpu.c`).
- The RVV ordinary kernels directly cast their inputs to `block_q2_K` and
  `block_iq4_nl`, load the original fields, and execute for VLEN 128/256.
  They do not require `block_q2_Kx16` or `block_iq4_nlx16`.
- `CPU_REPACK` is an extra buffer type. If it is not compiled or not enabled,
  the normal CPU buffer remains available and the ordinary type-table path is
  valid.
- The repack backend's `supports_op()` is conditional and returns false for
  unsupported shapes/types/operations, proving the system already models a
  fallback composition.

The source evidence proves that disabling repack selects the original
representation's ordinary RVV kernels, not a format-invalid path. A complete
same-artifact output-equivalence and performance run remains to be performed
on RV2. The baseline's successful output only proves the repacked path.

## E. Performance value

The committed evidence measures the enabled path only. It proves that repacking
is not the source of the vBuf startup advantage: vBuf repacking is about 8.44 s
slower at the warm median while total startup is about 26.53 s faster.

The requested with/without table is therefore:

| Metric | Repack enabled | Repack disabled |
|---|---:|---:|
| Startup -> ready | GGUF 101750.605 ms; vBuf 75107.327 ms | Not measured |
| First token | 0.000 ms event offset in harness | Not measured |
| Prompt evaluation | vBuf 1116.665 ms median | Not measured |
| Decode | 3-token smoke only; no steady-state rate claim | Not measured |
| Peak RSS | vBuf 7.282 GiB median | Not measured |
| CPU utilization / steady RSS / temporaries | Recorded only for enabled baseline | Not measured |

No claim is made that disabling repack is faster. The required experiment is a
separate RV2 binary with the same source, artifact, parameters, and sampler,
plus output comparison. This is `MORE_EVIDENCE_REQUIRED` for performance, not
for the architectural existence of a fallback.

## F. Scheduling constraints

The transformation itself has no dependency on the whole model. Each tensor
is repacked from one complete source tensor into one complete destination
tensor. Therefore, semantically it can be:

- tensor-at-a-time;
- layer-at-a-time for a bounded cache;
- first-use lazy;
- prefetched or asynchronously prepared before its operation;
- overlapped with unrelated I/O or previous-layer execution if ownership and
  synchronization are supplied.

The current llama/ggml composition makes it eager because model loading first
selects a buffer and allocates the complete model buffers, then
`load_all_data()` visits every tensor and calls `set_tensor()`, and only after
that does context/graph execution become ready. That is a loader ordering
constraint, not a mathematical or kernel correctness constraint.

One important limitation is that an individual `MUL_MAT_ID` operation needs
the transformed expert tensor before that operation begins. This prevents
freeing a transformed tensor while it is live, but does not require all 28
tensors to exist simultaneously.

## G. Memory ownership

For the benchmark's direct vBuf source, ownership is:

```text
vBuf consumer handle / mmap mapping
    |----------------------------- model lifetime -----------------------------|
        borrowed source tensor payload pointers
        |------------------------- model lifetime ------------------------------|

CPU backend buffer, 2171.14 MiB
    |----------------------------- model lifetime -----------------------------|

CPU_REPACK buffer, 2587.83 MiB
    |----------------------------- model lifetime -----------------------------|
```

The direct source retains the consumer handle until `llama_model_free_vbuf_direct`;
the callback attaches each tensor pointer from the mapping. The patched
user-tensor loading seam then copies user data into the destination buffer when
the callback changes `t->data`; for a CPU_REPACK tensor that destination is the
repacked CPU-backed buffer.

The measured allocated destination buffers are identical for GGUF and vBuf:
2171.14 MiB CPU plus 2587.83 MiB CPU_REPACK. Therefore they do not explain the
format difference. The vBuf process additionally retains the mapped source
file, whose pages are demand-faulted and may remain resident. The aggregate
RSS arithmetic is consistent with this:

```text
vBuf:  7.282 GiB RSS - (2.120 GiB CPU + 2.527 GiB REPACK)
       = about 2.635 GiB for mapped source pages and runtime state

GGUF:  4.780 GiB RSS - the same destinations
       = about 0.133 GiB for runtime state and sampling overhead
```

This is strong ownership evidence for the additional vBuf resident source
pages, but it is not a page-by-page proof that exactly 2.635 GiB is source
payload. The baseline did not instrument per-allocation RSS or `smaps` at each
phase. The exact peak decomposition is consequently `MORE_EVIDENCE_REQUIRED`.
The source mapping is not released after materialization because the direct
path intentionally uses its pointers for the model lifetime.

## H. Layer ownership

CPU_REPACK semantically belongs below the canonical execution representation:

```text
DirectExecution<original quant blocks>
    |
    +-- optional RepackExecution<RVV 16x1 kernels>
```

The CPU backend knows the kernel family, input type, VLEN, operation, shape,
destination ownership, and lifetime. The persistent format knows none of the
runtime kernel preference. The vBuf-ML profile should therefore not encode
`*_16x1` as a required representation merely to satisfy this current backend.

## I. vBuf constraints

For this model and direct path, the current conclusion is:

**CURRENT_VBUF_SUFFICIENT_BUT_RUNTIME_NEEDS_ADAPTER**.

The execution path already obtains the required type identity, dimensions,
exact byte length, independent tensor access, bounded physical ranges, and a
pointer whose lifetime is protected by the retained consumer handle. The vBuf
layout implementation validates payload alignment and bounded ranges. The
original representations satisfy the normal ggml block geometry, as proven by
the 377/377 payload parity and the successful direct/repacked execution.

No new format-level constraint is proven. A native runtime may still need an
adapter that converts descriptors into kernel-facing tensor views and owns the
mapping lifetime. That is runtime state, not a new vBuf physical requirement.

## J. Minimal native runtime implications

Starting from `mmap / RangeSource -> validated ModelView`, the demonstrated
minimum for this DeepSeek execution is:

| Component | Status | Reason |
|---|---|---|
| RangeSource/mmap and validated tensor descriptors | REQUIRED | Source bytes, bounds, type, shape, and lifetime are needed. |
| Architecture/model semantics | REQUIRED | Builds the DeepSeek layer operations and dimensions. |
| Direct tensor views | REQUIRED | Kernels need typed, addressable original blocks. |
| Matrix/vector kernels and dispatch | REQUIRED | Performs execution; direct IQ4_NL/Q2_K kernels already exist. |
| Temporary compute scratch | REQUIRED | Existing matrix operations quantize activations to Q8 variants and need workspace. |
| KV/cache/recurrent state | REQUIRED for generation | State is needed across tokens; not a model-payload concern. |
| Operation graph or an equivalent execution plan | REQUIRED | Orders dependencies and schedules kernels. |
| Tokenizer and sampling | REQUIRED for the demonstrated end-to-end product | Not required for a raw tensor-kernel executor. |
| Whole-model payload materialization | CURRENT_RUNTIME_ARTIFACT | The current ggml loader does this; direct views do not require it. |
| Eager CPU_REPACK | OPTIONAL | Required only by the selected optimized wrapper. |
| GGUF metadata synthesis | CURRENT_RUNTIME_ARTIFACT | Needed by this compatibility adapter, not by vBuf-native execution. |
| GGML tensor/context object model | CURRENT_RUNTIME_ARTIFACT | Useful adapter, not established as format or execution necessity. |

## K. Recommendation

The evidence supports these combined recommendations:

- **DIRECT_EXECUTION_ALREADY_POSSIBLE**: original IQ4_NL and Q2_K have ordinary
  RVV kernels with direct source-layout consumers.
- **KEEP_REPACK_AS_OPTIONAL_CPU_OPTIMIZATION** and **MOVE_REPACK_TO_BACKEND_WRAPPER**:
  its predicates and ownership are CPU/RVV kernel concerns. A composition such
  as `RepackExecution<DirectExecution>` preserves the purity rule.
- **MAKE_REPACK_LAZY**: eager whole-model creation is imposed by the current
  llama/ggml model loader, not by the transformation or kernels. A bounded
  tensor/layer cache is architecturally valid, subject to operation lifetime.
- **VBUF_NATIVE_RUNTIME_SHOULD_BYPASS_THIS_LAYER**: vBuf should expose and
  validate original tensor ranges; it should not inherit CPU_REPACK invariants.
- **CURRENT_GGML_RUNTIME_IS_ARCHITECTURALLY_INCOMPATIBLE** with the ideal vBuf
  residency model at this layer: it can be adapted, but its default buffer
  selection plus `load_all_data()` creates all transformed destinations before
  execution readiness and retains the source mapping through the adapter.
- **MORE_EVIDENCE_REQUIRED** only for the quantitative no-repack tradeoff and
  exact RSS page attribution. Those gaps do not justify a format change.

The next experiment should be a separate, reversible RV2 `GGML_CPU_REPACK=OFF`
build of the same custom benchmark. It should record output bytes/logits, the
same phase markers, sampler fields, and a dispatch marker proving use of
`ggml_vec_dot_iq4_nl_q8_0` / `ggml_vec_dot_q2_K_q8_K`. It should not alter the
committed evidence directory.
