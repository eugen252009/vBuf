# Step 11 qualification

Step 11 is an evidence step. No new tokenizer algorithm or portable cold-section
wire structure was selected.

## First-target result

The current plan does not pin a first real tokenizer adapter or model family
strongly enough to justify BPE, SentencePiece-like, WordPiece, merge tables,
normalizers, pre-tokenizers, or chat templates.

| Feature | Classification |
|---|---|
| VocabularyOnly | SUPPORTED FOR FIRST TARGET baseline |
| token text/offsets/scores/types | SUPPORTED FOR FIRST TARGET baseline |
| special-token IDs | SUPPORTED FOR FIRST TARGET baseline |
| BPE/merge tables | POSSIBLE BUT DEFERRED |
| SentencePiece-like behavior | POSSIBLE BUT DEFERRED |
| WordPiece | NOT REQUIRED / DEFERRED |
| chat templates | NOT REQUIRED / DEFERRED |
| tokenizer execution | runtime concern, deferred |

No merge representation was selected. If later required, numeric token-ordinal
arrays remain the preferred candidate over duplicated string pairs.

## Hot/cold model

The qualified baseline is:

```text
HOT AT MODEL OPEN
  canonical structure, Bootstrap, ModelMetadata, TensorDirectory

HOT BEFORE FIRST TOKENIZATION
  tokenizer control index and structural references

HOT ONLY WHEN TOKENIZATION IS REQUESTED
  TokenTextBytes, offsets, scores, types

COLD / OPTIONAL
  unrelated auxiliary values and future algorithm-specific structures
```

The current `TokenizerMetadata::parse` eagerly validates the complete UTF-8
text pool during tokenizer initialization. This is safe and deterministic but
means tokenizer initialization is not a metadata-only operation. Deferring
that content check is a future policy decision, not part of profile 0.1.

## Validation timing

- Canonical block geometry, ranges, padding: canonical open/validation.
- Bootstrap, model metadata, tensor directory references: semantic discovery.
- Tokenizer control structure, offsets, UTF-8 pool, and array consistency:
  tokenizer initialization in the current implementation.
- Token text access: zero-copy range access after initialization.
- Merge semantics: deferred because no merge structure is selected.

No arbitrary profile offsets are trusted. Large data remains canonical checked
ranges.

## Harness and methodology

`scripts/run_step11_qualification.sh` runs
`rust/vbuf-ml/examples/step11_qualification.rs`. The harness creates a
file-backed canonical v0.6 fixture containing:

- small bootstrap, model metadata, and tensor directory;
- a 4 MiB token pool;
- offsets, scores, and types arrays;
- an 8 MiB unrelated auxiliary region.

It maps the file read-only with `mmap` and reports phase time, process minor and
major fault deltas, explicitly touched bytes, and checksums. The recorded run
uses warm page-cache/process conditions. No `drop_caches` permission was
assumed, so fault counts are not storage-I/O claims. The benchmark deliberately
uses separate lazy and eager paths and avoids debug formatting of payloads.

The committed result is under `benchmark-results/vbuf-ml-step11/`.

Observed sample on the recorded host:

```text
lazy model/tensor path:       ~0.05 ms, no tokenizer payload access
initializer + token pool:    ~0.9 ms, ~4.2 MiB explicit tokenizer bytes
all token arrays:             ~0.1 ms additional, ~4.2 MiB explicit bytes
auxiliary cold range:         ~0.3 ms, 8 MiB explicit bytes
```

These are qualification measurements for this fixture and host, not universal
performance claims. The result supports the narrower claim that model-open
can stop before tokenizer and unrelated auxiliary ranges are accessed.

## Runtime-local derived indexes

No portable `LayerIndex`, `PhysicalRangeIndex`, or transfer plan was added.
The current profile does not define layer/group relationships independently of
future architecture semantics, so an ordinal layer index cannot yet be
qualified against a real target. Such an index remains a runtime-local
candidate:

```text
validated ModelMetadata + TensorDirectory
→ local semantic ordinals/ranges
```

Nano is not required. Any future Nano-assisted construction must produce the
same result as canonical descriptor derivation and must never authorize bytes.
Construction cost, lookup reuse, cold-start impact, and RAM overhead require a
real target before promotion can be justified.
