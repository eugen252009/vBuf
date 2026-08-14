# Step 19B: Qwen3 tokenizer qualification

Status: **raw tokenizer storage ready; consumer execution remains runtime-local**.

Pinned consumer:

```text
https://github.com/ggml-org/llama.cpp.git
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

## Consumer findings

| Source | Pinned symbol/path | Finding |
|---|---|---|
| `tokenizer.ggml.model` | `llama_vocab::impl::load` | required; `gpt2` selects BPE |
| `tokenizer.ggml.pre` | `llama_vocab::impl::load` | `qwen2` selects Qwen2 pre-tokenizer; `clean_spaces = false` |
| `tokenizer.ggml.tokens` | `llama_vocab::impl::load` | required vocabulary; ordinal is token ID |
| `tokenizer.ggml.token_type` | `llama_vocab::impl::load` | optional; values map to consumer token attributes |
| `tokenizer.ggml.scores` | `llama_vocab::impl::load` | optional; loaded for vocabulary metadata, not required by BPE execution |
| `tokenizer.ggml.merges` | `llama_vocab::impl::load` | required for GPT2 BPE; source order is rank |
| special ID keys | `llama_vocab::impl::load` | explicit values override model-family defaults |
| `tokenizer.ggml.add_bos_token` | `llama_vocab::impl::load` | optional consumer default is true; explicit false overrides it |
| `tokenizer.chat_template` | `llama_model_chat_template` | retrieved for chat formatting, not raw tokenization |

The source paths and exact matching lines are machine-recorded in
`benchmark-results/vbuf-ml-step19b/upstream-provenance.json`.

## Complete real-artifact inventory

Both Q8_0 and BF16 artifacts have the same inventory:

| Key | Type | Count/size | Classification |
|---|---|---:|---|
| `tokenizer.ggml.model` | string | 4 bytes | required raw-tokenizer identity |
| `tokenizer.ggml.pre` | string | 5 bytes | required raw-tokenizer identity |
| `tokenizer.ggml.tokens` | array[string] | 151,936 | required vocabulary |
| `tokenizer.ggml.token_type` | array[i32] | 151,936 | special-token behavior / attributes |
| `tokenizer.ggml.merges` | array[string] | 151,387 | required ranked BPE source |
| `tokenizer.ggml.bos_token_id` | u32 | 1 | special-token behavior |
| `tokenizer.ggml.eos_token_id` | u32 | 1 | special-token behavior |
| `tokenizer.ggml.padding_token_id` | u32 | 1 | special-token behavior |
| `tokenizer.ggml.add_bos_token` | bool | 1 | raw tokenization behavior |
| `tokenizer.chat_template` | string | exact UTF-8 bytes | chat-template parity only |

Machine-readable inventory is in `tokenizer-source-matrix.csv`.

## Merge qualification

The pinned loader uses:

```cpp
const size_t pos = word.find(' ', 1);
first  = word.substr(0, pos);
second = word.substr(pos + 1);
bpe_ranks.emplace(std::make_pair(first, second), i);
```

For both artifacts:

```text
merge count:       151,387
vocabulary count:  151,936
malformed merges:  0
missing components: 0
duplicate pairs:   0
```

The selected portable representation is two u32 arrays of vocabulary ordinals:

```text
MergeLeftIds[rank]
MergeRightIds[rank]
```

The rank is the array ordinal. No lexical sorting or rank rewriting occurs.
The runtime adapter reconstructs the pinned consumer’s string-keyed lookup from
existing direct vocabulary ranges. It does not serialize a hash table.

Comparison evidence is in `merge-representation-comparison.csv`.

## Requirement classification

| Semantic | Classification |
|---|---|
| GPT2 BPE identity | `REQUIRED_FOR_RAW_TOKENIZATION_PARITY` |
| Qwen2 pre-tokenizer identity | `REQUIRED_FOR_RAW_TOKENIZATION_PARITY` |
| vocabulary text and ordinals | `REQUIRED_FOR_RAW_TOKENIZATION_PARITY` |
| ranked merges | `REQUIRED_FOR_RAW_TOKENIZATION_PARITY` |
| token types | `REQUIRED_FOR_SPECIAL_TOKEN_BEHAVIOR`; optional to BPE execution |
| BOS/EOS/PAD IDs | `REQUIRED_FOR_SPECIAL_TOKEN_BEHAVIOR` |
| `add_bos` | `REQUIRED_FOR_RAW_TOKENIZATION_PARITY` |
| scores | `OPTIONAL_FOR_FIRST_RAW_INFERENCE_TARGET` |
| chat template bytes | `REQUIRED_FOR_CHAT_TEMPLATE_PARITY`, not first raw target |
| add-EOS/add-prefix/normalizer flags | `NOT_REQUIRED_BY_PINNED_CONSUMER` for these artifacts; absent |

## Portable profile

Added tokenizer kind `Gpt2BpeQwen2` while preserving `VocabularyOnly`.
Profile-local identities are:

```text
TokenizerModel::Gpt2Bpe = 1
PreTokenizer::Qwen2    = 1
```

Added roles for merge arrays, identities, `AddBos`, and optional
`ChatTemplate`. The control structure remains small; large merge arrays and
chat bytes remain separate canonical regions.

No execution methods or llama.cpp dependencies were added.

## Lazy loading

```text
model open
  → canonical validation, bootstrap, model metadata, tensor directory
  → no vocabulary, merge, or chat payload access

tokenizer initialization
  → vocabulary and merge range resolution/validation
  → runtime may build its derived BPE lookup

chat formatting request
  → optional chat-template range access
```

Structural range authorization remains separate from semantic merge validation.
No unrelated payload is scanned at model open.

## Parity result

Both artifacts match for vocabulary, token types, merge source and resolved
ordinal pairs, identities, special IDs, `add_bos`, and chat-template bytes.
Evidence is in `tokenizer-parity.csv`.

No independent tokenizer implementation was used, so actual token-ID output
parity is not claimed until the later pinned-consumer adapter. This step proves
that all source semantics needed to reconstruct that consumer configuration are
represented losslessly.

## Readiness

```text
raw inference tokenizer: READY
chat template storage: READY
chat execution parity: DEFERRED TO CONSUMER
conversion: READY_FOR_CONVERSION
```
