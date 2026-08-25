# Step 32A: Streaming Hugging Face Safetensors -> vBuf

## Scope

This step adds a vBuf-ML-owned importer at `rust/vbuf-ml/src/hf_import.rs` and
the `vbuf-ml-import-hf` CLI. It does not modify v0.6, GGML, llama.cpp, runtime
residency, model semantics, or serving concurrency.

The existing local conversion path remains the qualified manifest-driven
GGUF-to-vBuf writer in `scripts/convert_gguf_to_vbuf_ml.py` and
`vbuf-ml-convert`. It maps a complete local source through the sequential
canonical writer. Step 32A adds a separate execution mechanism but reuses the
same `LayoutPlan` geometry and the existing bootstrap, metadata, and tensor
directory encoders.

## Architecture

```text
HF repository metadata and pinned revision
    -> Safetensors index and bounded shard headers
    -> SourceTensor records
    -> LayoutPlan::build_lengths
    -> pre-sized model.vbuf.partial
    -> bounded HTTP ranges and destination scatter writes
    -> model.vbuf
```

Planning does not read tensor payload bytes. Every destination block start,
payload offset, length, occurrence, and alignment is known before the first
bulk payload range. v0.6 `BaseShift = 4` and its 16-byte BaseStep are used; no
new page or backend alignment rule was introduced.

Safetensors offsets are interpreted as required:

```text
data_start = 8 + header_length
source_start = data_start + data_offsets[0]
source_end = data_start + data_offsets[1]
```

All source, destination, range, alignment, padding, and aggregate accounting
uses checked `u64` arithmetic. The `PlacementLengthRequest` extension is the
shared planning half of the existing profile layout planner, not a second
remote-specific format.

## Storage And Lifetime Invariants

- `model.vbuf.partial` is pre-sized to its final canonical size and is the
  eventual output, not a source cache.
- Tensor windows are bounded by the configured staging limit (64 MiB by
  default) and are scattered with positioned writes directly into final tensor
  payload offsets.
- Adjacent source tensor ranges may be coalesced, subject to the staging limit
  and a 64 KiB gap limit.
- The source bulk byte count written to disk is zero.
- No complete Safetensors shard, payload temporary, second vBuf, or GGUF
  intermediate is created.
- The journal is metadata-scale JSON and records plan digest, source identity,
  final size, and completed tensor names. The transition is durable destination
  write, `sync_data`, then journal publication. Journal updates use temporary
  metadata and rename.
- A partial artifact without its state file, a changed source identity, a
  changed plan digest, or an existing final output is rejected.
- Completion renames the partial output only after canonical v0.6 parsing,
  bootstrap discovery, and tensor-directory validation.

The implementation uses bounded ureq response-body limits. A range response
that is not `206`, has the wrong `Content-Range`, is short, or exceeds its
requested bound is rejected. Redirects are followed manually so the range is
re-applied to the signed storage URL and bearer credentials are not sent to a
different host.

## Metadata And Semantics

The planner fetches `config.json`, the optional Safetensors index, and optional
generation/tokenizer metadata files without executing repository code. It does
not use Transformers, `from_pretrained`, `trust_remote_code`, or Python model
code. `HF_TOKEN` is read only by the transport and is not placed in a plan,
journal, vBuf artifact, diagnostic, or source file.

The index is cross-checked against every actual shard header. Missing indexed
tensors, unindexed header tensors, duplicate ownership, and shard disagreement
fail closed. Supported storage mappings in this step are exact F32
`CanonicalPrimitive` bytes and BF16 opaque bytes. Other Safetensors dtypes are
reported as storage descriptor gaps; no cast or quantization is performed.

The current profile can persist the declarative architecture and core numeric
configuration fields. Tokenizer persistence and new architecture-specific
expert-bank derivation are explicitly reported as runtime/semantic gaps rather
than guessed from names. Dense storage conversion and backend execution are
separate qualifications.

## Offline Evidence

`rust/vbuf-ml/tests/hf_import.rs` and module tests pass for:

- Safetensors header length, UTF-8/JSON, shape/offset, overlap, and exact
  source-offset derivation;
- index ownership agreement and disagreement;
- canonical aligned destination placement and exact tensor byte parity;
- direct destination scatter from a bounded source response;
- bounded retry after a connection failure;
- interrupted conversion leaving `.partial` and `.convert-state`, followed by
  resume without a second model file;
- final output publication and source-revision mismatch rejection;
- insufficient-space gate;
- synthetic logical plans at 4 GiB, 64 GiB, 128 GiB, and 256 GiB without
  allocating those byte ranges.

The local fixture uses one F32 tensor with an 8-byte payload. Its payload bytes
are written exactly at the planned canonical destination range and the final
artifact is reopened through the normal v0.6, bootstrap, and tensor-directory
readers.

## Real HF Status

The public repository `hf-internal-testing/tiny-random-LlamaForCausalLM` at
resolved revision `9fb191250dd56d0ba7ec9785a025ed29c03d5998` completed both
plan-only and full single-file conversion qualification. The plan reported 21
tensors, one shard, `4,129,088` source payload bytes, `4,131,200` final vBuf
bytes, `334` alignment/padding bytes, 16-byte destination alignment, and one
coalesced payload request. Planning fetched 7 bounded metadata/header
responses and `1,848,263` bytes. The full conversion wrote `4,129,088`
destination bytes, published the canonical artifact, and left no complete
source shard or second model payload. The final artifact reopened through
canonical v0.6, bootstrap, and tensor-directory validation.

The manageable public sharded smoke was not run because no suitable small
sharded public repository was selected during this step. Sharded public
qualification remains open.

## Accounting Boundary

Offline tests establish the accounting fields and invariants, but no large
model measurement is claimed. For a real run the CLI reports planning requests,
planning bytes, source payload bytes, final vBuf bytes, padding, staging limit,
free-space gate, payload request/return bytes, resume counters, and destination
writes. RSS, final validation reads, and additional storage must be measured by
the qualification harness around the CLI; they are not inferred from elapsed
time.

## Readiness

```text
VBUF_STORAGE_READY: YES for the supported exact-byte F32/BF16 storage profile
MODEL_RUNTIME_READY: NO for tokenizer/architecture gaps not persisted here
BACKEND_EXECUTION_READY: NO claim; backend qualification is a separate boundary
REAL_HF_PLAN_ONLY_READY: YES for a pinned metadata-only storage plan; sharded
  public smoke remains open before bulk conversion
```

The next authorized step should qualify a manageable sharded public repository
through the same bounded redirect/range transport, then run the separately
authorized 100--200+ GiB plan-only inspection. It must remain plan-only before
any large-model transfer.
