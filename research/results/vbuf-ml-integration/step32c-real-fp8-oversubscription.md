# Step 32C: Real FP8 model above measured host plus GPU memory

## Scope

This qualification exercises the vBuf-ML-owned Hugging Face Safetensors
planner and direct range executor with a pinned public FP8 MoE model. It does
not change v0.6, GGML, llama.cpp, backend loader ownership, or production
serving concurrency. The generated model artifact was a qualification output
and was not committed.

## Environment and gates

```text
BRANCH: vbuf-ml
TARGET: Linux x86_64
PHYSICAL_RAM_BYTES: 67304775680
GPU0: NVIDIA GeForce RTX 3060, 12884901888 bytes VRAM
GPU1: NVIDIA GeForce RTX 2080 SUPER, 8589934592 bytes VRAM
TOTAL_GPU_VRAM_BYTES: 21474836480
RAM_PLUS_GPU_VRAM_BYTES: 88779612160
TARGET_FILESYSTEM: /dev/nvme0n1p1, ext4
INITIAL_FREE_BYTES_BEFORE_PREALLOCATION: 241596993536
```

The selected source was `zai-org/GLM-4.5-Air-FP8` at immutable revision
`f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`.

```text
SAFETENSORS_SHARDS: 47
TENSORS: 36323
SOURCE_PAYLOAD_BYTES: 112558098944
FINAL_VBUF_BYTES: 112563538898
LARGEST_TENSOR_BYTES: 1241513984
STAGING_BYTES: 2147483648
PARALLEL_RANGE_REQUESTS: 4
SPACE_GATE: PASS
CAN_EXECUTE: YES
```

The source payload exceeded measured RAM plus GPU VRAM by
`23778486784` bytes. No local Safetensors shard or complete source payload
copy was created.

## Plan and conversion

Planning fetched 99 bounded metadata/header responses and 27,877,034 bytes.
All 36,323 destination offsets were known before payload acquisition.

```text
DTYPE_DISTRIBUTION: BF16=289, F32=18040, F8_E4M3=17994
FP8_QUANTIZATION_ENTRIES: 17994
DESTINATION_ALIGNMENT_BYTES: 16
ALIGNMENT_PADDING_BYTES: 146420
PAYLOAD_OVERFETCH_BYTES: 0
```

The first serial attempt durably completed 18,945,218,560 source bytes and
was stopped at its journal boundary. The same partial artifact was resumed
with four concurrent, independently bounded direct ranges. The resumed phase
reported:

```text
PAYLOAD_HTTP_REQUESTS: 79
PAYLOAD_REQUESTED_BYTES: 93612880384
PAYLOAD_RETURNED_BYTES: 93612880384
DESTINATION_WRITTEN_BYTES: 93612880384
RESUME_SKIPPED_BYTES: 18945218560
RESUME_DOWNLOADED_BYTES: 93612880384
SOURCE_OVERFETCH_BYTES: 0
```

The final artifact was pre-sized and ranges were scattered directly into its
canonical tensor offsets. Completion renamed the partial artifact only after
canonical validation. The highest sampled importer RSS during the parallel
run was approximately 3,538,000 KiB; GPU memory remained at ambient levels.

## Reopen and materialization

Fresh low-level reopen passed for canonical v0.6 parsing, bootstrap discovery,
model metadata, tensor directory, and FP8 quantization-directory parsing.
The artifact contained 17,994 FP8 tensors and 17,994 matching provenance
entries. Three FP8 tensors at early, middle, and late directory positions were
materialized through the bounded E4M3FN path:

```text
SAMPLED_FP8_TENSORS: 3
MATERIALIZED_F32_ELEMENTS: 99352576
MATERIALIZATION_VALUES_FINITE: YES
MATERIALIZATION_MAX_RSS_KIB: 1523628
```

The architecture is `Glm4MoeForCausalLM`. The plan explicitly reports
`expert-bank provenance requires an architecture profile`; no MoE expert
directory or guessed expert-bank mapping was emitted. The Safetensors index
also reported a `total_size` different from header-derived payload bytes;
headers remained authoritative and the discrepancy was retained as a
diagnostic warning.

Tokenizer metadata files were fetched during planning, but this importer does
not yet persist a tokenizer region. Consequently the existing full
`ConsumerModel` open path remains intentionally not runtime-ready and reports
the missing tokenizer role. This qualification claims storage reopen and
bounded FP8 materialization only, not tokenizer, MoE execution, generation, or
backend qualification.

During the first fresh reopen, a control-header semantic mapping defect was
found: `Signed` and `Float` discriminants were reversed while initializing
resumed artifacts. The mapping was fixed in `hf_import.rs`, a fixture now
reopens and parses floating model metadata, and the already-downloaded final
artifact was repaired at its two floating metadata anchors without redownloading
weights. Fresh reopen and materialization then passed.

## Verification

```text
cargo test -p vbuf-ml --test hf_import: PASS, 9 tests
cargo test -p vbuf-ml --lib: PASS, 10 tests
cargo check -p vbuf-ml --bin vbuf-ml-import-hf: PASS
cargo fmt --all: PASS
git diff --check: PASS
```

The implementation change and this report are committed together. The commit
does not include the generated vBuf artifact, source weights, credentials, or
qualification logs.

## Qualification boundary

```text
REAL_PINNED_FP8_PLAN: PASS
DIRECT_RANGE_TO_FINAL_VBUF: PASS
OVERSUBSCRIPTION_STORAGE_GATE: PASS
NO_FULL_SOURCE_COPY: PASS
PERSISTENT_FP8_PROVENANCE: PASS
BOUNDED_FP8_MATERIALIZATION: PASS
TOKENIZER_PERSISTENCE: OPEN
MOE_EXPERT_BANK_SEMANTICS: OPEN
BACKEND_EXECUTION: NOT CLAIMED
GENERATION_PARITY: NOT RUN
```
