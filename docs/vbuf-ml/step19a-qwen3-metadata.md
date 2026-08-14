# Step 19A: resolve pinned Qwen3 model-metadata gaps

Status: **qualified; model metadata ready, tokenizer remains blocked**.

## Pinned authority

```text
https://github.com/ggml-org/llama.cpp.git
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

Relevant pinned behavior:

- `llama_model.cpp` loads `attention.head_count_kv` into the per-layer
  `n_head_kv_arr`, defaulting it to `head_count` only when absent;
- `n_embd_head_k_full` and `n_embd_head_v_full` initially default to
  `embedding_length / head_count`, but the consumer separately overrides them
  from `attention.key_length` and `attention.value_length` when present;
- the graph uses key and value head dimensions independently.

Exact source locations and snippets are recorded in
`benchmark-results/vbuf-ml-step19a/upstream-provenance.json`.

## Source metadata

Both artifacts contain the same values:

| Source key | Q8_0 | BF16 | Consumer semantic |
|---|---:|---:|---|
| `qwen3.attention.head_count` | 16 | 16 | HeadCount |
| `qwen3.attention.head_count_kv` | 8 | 8 | KVHeadCount |
| `qwen3.attention.key_length` | 128 | 128 | KeyHeadDimension |
| `qwen3.attention.value_length` | 128 | 128 | ValueHeadDimension |
| `qwen3.embedding_length` | 1024 | 1024 | EmbeddingLength |

## Decisions

### KV head count

`KVHeadCount` is directly stored and independently consumed. It is not derived
from `HeadCount`.

### Key head dimension

`KeyHeadDimension` is directly stored and independently consumed. The pinned
consumer has a fallback of `EmbeddingLength / HeadCount`, but the explicit GGUF
key overrides that fallback. It is therefore not safe to discard the source
value.

### Value head dimension

`ValueHeadDimension` is independently stored and consumed. It currently equals
`KeyHeadDimension`, but the contract preserves the distinction.

## Portable metadata

Added optional profile-local ModelMetadata keys:

```text
9  = KVHeadCount
10 = KeyHeadDimension
11 = ValueHeadDimension
```

All use existing canonical unsigned scalar storage. They are optional for the
general profile and may be required by a consumer-specific Qwen3 qualification.
The profile version remains `0.1` because the metadata index already supports
known optional keys and unknown optional entries remain non-fatal.

The portable parser does not enforce Qwen3-specific presence. That requirement
stays at the consumer/qualification boundary.

## Real-artifact parity

The Step 19A qualification matched all ten values across both artifacts:

```text
2 artifacts × 5 semantic values = 10 matches
```

Evidence:

```text
benchmark-results/vbuf-ml-step19a/metadata-parity.csv
benchmark-results/vbuf-ml-step19a/metadata-source-matrix.csv
```

## Step-18 manifest result

The prior `MISSING` statuses are now `DIRECTLY_REPRESENTED`.

Both manifests now report:

```text
manifest_structurally_valid = true
metadata_blockers = []
conversion_readiness = BLOCKED_BY_TOKENIZER_GAP
```

No conversion writer or model bytes were produced.

## Remaining blocker

Tokenizer semantics remain unchanged and blocked for Step 19B. This step did not
interpret GPT-2/BPE model data, merges, pre-tokenizer behavior, or chat
templates.
