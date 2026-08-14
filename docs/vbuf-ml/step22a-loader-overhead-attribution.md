# Step 22A: vBuf loader overhead attribution

Status: **measured; no optimization introduced**.

The immutable Step-22 baseline remains under
`benchmark-results/vbuf-ml-step22/`. Step 22A uses a separate diagnostic tree:
`benchmark-results/vbuf-ml-step22a/`.

## Method

The diagnostic verifies the immutable consumer checkpoint
`vbuf-ml-0.1-consumer-parity` (`73a1f36661482037573a789ab90e15039a782ad9`),
the pinned llama.cpp commit, the adapter patch, and all four artifact hashes.
It runs ten warm samples per format/artifact.

The vBuf Rust diagnostic measures:

```text
mmap/open
canonical v0.6 validation
Bootstrap
ModelMetadata
TensorDirectory
TokenizerMetadata
ConsumerModel open/materialization
```

The C++ diagnostic measures:

```text
Rust consumer open
C++ tensor inventory
GGUF-compatible metadata synthesis
vocabulary projection
numeric-merge reconstruction
tensor metadata projection
llama_model_init_from_user
payload attachment callback
```

A native GGUF control executable measures GGUF metadata/tensor-directory parsing
without constructing a llama model. Native full model construction is measured
by the same pinned runtime probe; its internal phases are not split because no
instrumentation patch was added to the native GGUF path.

## Results

Representative ten-run medians:

| Artifact | GGUF model-ready | vBuf model-ready | Delta |
|---|---:|---:|---:|
| BF16 | 249.0 ms | 413.1 ms | +164.0 ms |
| Q8_0 | 194.6 ms | 419.1 ms | +224.5 ms |

Format-only parsing was small relative to the total:

```text
BF16 GGUF parse: approximately 21 ms
BF16 vBuf canonical/ML parse: approximately 38 ms
Q8_0 results are similar
```

The vBuf C++ trace showed:

```text
Rust ConsumerModel open/materialization: approximately 132 ms
GGUF-compatible metadata synthesis: approximately 132 ms
  vocabulary projection: approximately 55 ms
  merge projection: approximately 75 ms
Tensor metadata projection: approximately 0.6 ms
C++ tensor inventory: approximately 0.18 ms
llama_model_init_from_user: approximately 147–152 ms
payload attachment callback: approximately 57 us
```

The adapter performed approximately:

```text
910,600 C ABI calls
151,936 vocabulary entries
151,387 merge entries
311/310 tensor callbacks
```

The exact per-run data is in `phase-summary.csv`, `ffi-summary.csv`, and
`runtime-summary.csv`.

## Hypothesis results

| Hypothesis | Result |
|---|---|
| H1: canonical vBuf parsing dominates | Rejected; canonical validation was about 0.8 ms |
| H2: descriptor/FFI projection is material | Confirmed; consumer materialization was about 132 ms |
| H3: GGUF-compatible metadata synthesis dominates | Confirmed; about 132 ms |
| H4: merge reconstruction dominates | Partially confirmed; about 75 ms, the largest synthesis subphase |
| H5: `llama_model_init_from_user` is intrinsically slower | Not supported; its measured region was not larger than the inferred native remainder |
| H6: tensor bookkeeping dominates | Rejected; tensor projection was sub-millisecond and callbacks were tens of microseconds |
| H7: phase boundary was misleading | Rejected for warm attribution; explicit phases account for the vBuf total |
| H8: one-sample metadata result was noise | Rejected; ten warm samples reproduce the overhead |

The current path does duplicate substantial semantic work:

```text
vBuf mmap
→ Rust ConsumerModel snapshot/materialization
→ C ABI token/tensor projections
→ C++ owned tokenizer/merge strings
→ GGUF-compatible in-memory metadata
→ llama_model_init_from_user
```

The Rust consumer open and C++ projection are separate measured stages; the
current C ABI is fine-grained and performs roughly 910k calls for Qwen3.

## Native versus adapter cost

The evidence supports this classification:

```text
INTRINSIC_TO_VBUF:
  approximately 0.8 ms canonical validation plus approximately 1.1 ms directory parsing

CURRENT_ADAPTER_OVERHEAD:
  ConsumerModel materialization
  fine-grained vocabulary/token access
  string vocabulary and merge reconstruction
  GGUF-compatible metadata synthesis

LLAMA_USER_SOURCE_COST:
  not shown to be the dominant cost; tensor registration and payload callbacks are small

COMMON_RUNTIME_COST:
  llama model architecture/tensor construction and backend setup
```

The native GGUF internal phase split remains a limitation. The native full-load
measurement and format-only control establish the comparison boundary, but do
not provide per-function GGUF timing without adding a separate diagnostic patch.
Metadata lookup/allocation counters were not instrumented; allocation/copy
results therefore report exact payload-copy behavior and observed FFI string
bytes, but not general allocator counts.

## First-touch observation

No new uncached run was used for attribution. The existing Step-22 uncached
approximation remains the reference: vBuf tends to defer more mapped-page work
until prompt evaluation, while warm model construction shows CPU-side adapter
churn. The combined model-ready plus first-prompt figures remain separate from
this warm attribution and are not reclassified as cold-cache proof.

## Conclusion

The warm vBuf overhead is not primarily canonical vBuf parsing. It is primarily
the current prototype's repeated descriptor/materialization and consumer-local
GGUF-compatible tokenizer metadata construction, especially the 151,387-merge
string reconstruction. This is an adapter architecture finding, not a format
change recommendation.

No direct-native source seam, ABI redesign, merge representation change,
pre-fetch, partial loading, layout change, or other optimization was
implemented. Those are Step-23 candidates only.
