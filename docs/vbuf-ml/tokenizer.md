# vBuf-ML tokenizer metadata (Step 10)

Tokenizer metadata is an optional, independently loadable semantic index. It
is separate from model metadata and TensorDirectory.

Profile 0.1 defines `VocabularyOnly` kind. It provides a direct vocabulary view
but does not implement tokenization algorithms or merge execution. BPE,
SentencePiece-like, WordPiece, merges, chat templates, and added-token
algorithms remain future work.

## Composition

The index references canonical values:

```text
TokenTextBytes → opaque UTF-8 byte pool
TokenOffsets   → u64 array with vocabulary_count + 1 values
TokenScores    → optional f32/f64 array
TokenTypes     → optional unsigned integer array
special IDs    → optional unsigned scalar values
```

Token IDs are implicit ordinals. For token `i`, its text is delimited by
`offsets[i]..offsets[i+1]`. No token objects or token-ID array are materialized.
The parser validates UTF-8 and rejects embedded NUL bytes; empty token strings
are permitted.

All arrays must have consistent vocabulary length. Special IDs must be within
the vocabulary. Large text and array values remain canonical checked ranges and
are exposed through direct accessors.

## Wire format

Payload magic is `VBTOK\0\0\0`, version `1`, little-endian.

Header: 20 bytes:

```text
magic[8], version:u16, flags:u16, kind:u8, reserved:u8,
entry_count:u16, reserved:u32
```

Each 12-byte entry contains:

```text
role ID:u16
required flag:u16
canonical Key-ID:u16
canonical physical occurrence:u16
reserved:u32
```

Roles 1 and 2 (`TokenTextBytes`, `TokenOffsets`) are required. Scores, types,
and BOS/EOS/UNK/PAD IDs are optional. Entries are sorted by role ID and
duplicates fail. The control region is bounded to 4096 bytes and 256 entries;
the referenced text pool is bounded to 64 MiB and each token text to 4096
bytes. Vocabulary size is bounded to 10,000,000.

No offsets are trusted as physical file offsets: tokenizer offsets are domain
indices into the separately referenced canonical byte pool and are validated
against that pool.
