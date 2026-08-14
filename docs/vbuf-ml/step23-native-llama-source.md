# Step 23: source-neutral llama model construction

Status: **qualified for Qwen3 BF16 and Q8_0 on CPU**.

Step 22/22A evidence is unchanged. New evidence is under
`benchmark-results/vbuf-ml-step23/`.

## Pinned pipeline and gap report

The pinned llama.cpp path is:

```text
llama_model_load_from_file
→ llama_model_loader (GGUF metadata, tensor inventory, mmap weights)
→ llama_model_create
→ llama_prepare_model_devices
→ llama_model_base::load_hparams
→ llama_model_base::load_vocab
→ llama_model_base::load_tensors
→ GGML backend buffers
→ common graph/decode runtime
```

The format-specific coupling was concentrated in `llama_model_loader` and
`llama_vocab::impl::load`:

- typed metadata used `gguf_context` lookups;
- tokenizer tokens, scores, types, and merges were read from GGUF arrays;
- tensor existence/type/shape was resolved through GGUF tensor entries;
- the existing user-source hook still required an in-memory GGUF context.

The source-neutral convergence point is now the loader-to-model-base boundary.
GGUF retains its existing native parser and loader path. A non-GGUF source
enters the same `llama_model_base` hparams, vocabulary, tensor, backend, graph,
and decode construction after supplying semantic loader data.

## Old compatibility path

```text
vBuf mmap
→ ConsumerModel owned snapshot
→ fine-grained C ABI
→ C++ owned tokenizer/merge structures
→ GGUF-compatible metadata context
→ llama_model_init_from_user
```

## New direct path

```text
vBuf mmap
→ validated Rust ConsumerModel bulk views
→ llama_model_source semantic interface
→ common llama_model_loader/model_base builder
→ llama_model
```

The direct path does not create a `gguf_context`, does not synthesize GGUF
metadata, and does not insert GGUF merge strings.

## Source interface

`src/llama-model-source.h` introduces the smallest pinned-runtime seam used by
the experiment:

```text
required/optional typed metadata lookup
bulk tensor descriptor lookup
borrowed tensor payload pointer/length
bulk token view
bulk numeric merge view
```

The interface contains no vBuf or GGUF types other than the runtime-neutral
`ggml_type` and tensor dimensions required to construct GGML metadata.

The GGUF path remains unchanged semantically and continues to use the existing
GGUF loader. The new interface is used only by the direct external source.

## Metadata

The vBuf source supplies the required Qwen3 semantic keys through typed lookup:

```text
architecture
context length
embedding length
layer count
head count / KV head count
key/value head dimensions
feed-forward length
RMS epsilon
RoPE theta
tokenizer model/pre-tokenizer
special token IDs
BOS behavior
chat template
```

No Qwen3 values are embedded in the generic source seam. The vBuf source maps
validated vBuf-ML metadata fields to the generic keys.

## Tokenizer

The Rust FFI now exposes immutable bulk views:

```text
TokenView[]: text pointer/length, score, type
MergeView[]: left token ID, right token ID
```

The direct tokenizer builder consumes numeric merge IDs and resolves token text
only at the final llama BPE-map construction boundary. It never creates
GGUF merge-array strings or a temporary GGUF metadata array.

Final llama runtime BPE structures still own the strings they fundamentally
require; this is one final runtime representation, not a GGUF intermediate.

## Tensor source

The direct source consumes one immutable bulk tensor table:

```text
name pointer/length
representation
rank/dimensions
borrowed payload pointer/length
```

Mappings are preserved:

```text
F32  → GGML_TYPE_F32
BF16 → GGML_TYPE_BF16
Q8_0 → GGML_TYPE_Q8_0
```

Payload copy, repack, and byte reorder remain zero.

For Q8_0, missing `output.weight` uses the existing tied-output behavior by
referencing `token_embd.weight`. BF16 retains explicit `output.weight`.

## Ownership and lifetime

The direct integration retains a `shared_ptr<llama_model_source>` keyed by the
returned `llama_model`. The Rust consumer handle and mmap therefore outlive
model construction and all model/tokenizer use. `llama_model_free_vbuf_direct`
first frees the llama model and then releases the source ownership.

Failure during construction releases the temporary shared source. The direct
payload callback rejects missing names, type mismatches, and byte-length
mismatches before assigning the borrowed pointer.

## ABI reduction

Measured call counts:

| Path | BF16 | Q8_0 |
|---|---:|---:|
| compatibility adapter | 910,601 | 910,599 |
| direct source | 12 coarse calls | 12 coarse calls |

The direct calls are open, architecture/metadata, three bulk tables, four
special-token lookups, BOS, and chat-template retrieval. Token and merge
entries do not cross the ABI individually.

## Warm model-ready results

Ten warm samples per artifact/path, CPU-only, diagnostic tracing disabled:

| Artifact | GGUF | Compatibility | Direct |
|---|---:|---:|---:|
| BF16 | 249.3 ms | 414.1 ms | 247.1 ms |
| Q8_0 | 196.0 ms | 416.3 ms | 247.0 ms |

The direct path removes nearly all of the measured compatibility adapter delta:

```text
BF16 old vBuf overhead: ≈164.8 ms
BF16 direct overhead:   ≈-2.2 ms over GGUF

Q8_0 old vBuf overhead: ≈220.3 ms
Q8_0 direct overhead:    ≈51.0 ms over GGUF
```

The remaining Q8_0 difference is not attributed to a new tensor copy or
representation conversion. It remains a source/runtime construction and
measurement boundary for later investigation.

## Correctness

Three-way qualification was run against native GGUF, the Step-21 compatibility
adapter, and the direct source.

For both BF16 and Q8_0:

```text
metadata parity: PASS
vocabulary/tokenizer parity: PASS
special-token and chat-template parity: PASS
logit max_abs_diff: 0
8-token deterministic generation: PASS
```

Observed deterministic output for all three paths:

```text
0,1096,374,264,4285,3110,315,264
```

The existing Step-21 payload digest and structural evidence remains the oracle;
the direct path uses the same validated bulk tensor descriptors and performs
additional runtime type/size checks before attaching every payload.

## GGUF regression

Native GGUF loading, tokenizer construction, logits, and deterministic
generation continued to pass through the same pinned native path. The direct
source changes are guarded by the source object and do not add source branches
to graph construction, kernels, scheduler, decode, or generation.

## Patch series

```text
patches/llama.cpp/0001-user-metadata-tensor-source.patch
patches/llama.cpp/0002-source-neutral-model-source.patch
```

The second patch adds the source seam and direct loader path while preserving
the first patch for historical Step-21 reproduction.

## Limitations

- The GGUF adapter is a thin conceptual source wrapper only at the common
  loader boundary; the native GGUF parser remains unchanged rather than being
  rewritten around the interface.
- Rust `ConsumerModel::open` still materializes its validated snapshot. The
  direct path eliminates the second C++/GGUF reconstruction, but eliminating
  that Rust snapshot is a later borrowed-view experiment.
- Payload callback lookup is currently a simple source lookup and is not a
  performance-critical measured phase.
- ASan/UBSan end-to-end qualification was not run against the prebuilt pinned
  llama.cpp shared libraries; the source seam was compile-qualified and Rust
  Clippy/tests pass.
- GPU, cold-cache, first-touch, prefetch, and inference-kernel behavior remain
  deferred.

## Conclusion

The pinned llama.cpp structure permits a clean source-neutral model-construction
seam without changing the vBuf wire format or the common runtime. vBuf-ML can
now be a first-class semantic source rather than pretending to be GGUF.

The measured representation churn identified in Step 22A was removed from the
direct path, while the Step-21 compatibility adapter remains available as the
migration oracle.
