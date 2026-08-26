# Step 32H: Real Text To Full-Stack Logits

Date: 2026-08-26

## Result

`REAL_PERSISTED_TEXT_TO_LOGITS_PASS_WITH_ORDERED_ROUTE_NEAR_TIE_NOTE`

The generic portable Rust runner accepted the fixed UTF-8 text `Test`, used
the persisted tokenizer, gathered its embedding rows directly from the
authoritative payload, executed the complete persisted base stack `0..45`,
ran final RMSNorm, and projected logits through the persisted LM head. No
decode, generation, GGML, device, or backend loader path was used.

The independent bounded NumPy reference reproduced the recorded token IDs and
embedding rows exactly. Its full propagated replay matched the final-position
argmax and top-10. Two ordered Top-8 slots differed at layer `24`, token `3`:
experts `54` and `104` exchanged ranks. The selected expert set was identical
at both slots, so there were zero routing membership mismatches. The result is
qualified with the existing numerical ordered-route near-tie boundary; strict
full-chain ordered route-ID equality is not claimed.

## Qualified Input

- Text: `Test`
- UTF-8 bytes: `4`
- Text SHA-256: `532eaabd9574880dbf76b9b8cc00832c20a6ec113d682299550d7a6e0f345e25`
- Persisted tokenizer: GPT-2 BPE byte-level
- Persisted pre-tokenizer: GPT-2 byte-level
- Added special tokens: none
- Token IDs: `[51, 68, 82, 83]`
- Token ID hash: `68d6ef8dc632209d60b1f653bace515d4c17d13ef47ff4ae0d3d35d78107bb0f`
- Tokenizer round trip: exact
- Token-ID fixture reference: pass

The production path invokes `Gpt2ByteLevelTokenizer` over the persisted
sidecar. The numerical reference consumes the recorded IDs and independently
reads the corresponding persisted BF16 rows; its tokenizer check is the
captured independent fixture for this fixed qualification input, not a second
Rust tokenizer invocation.

## Embedding And Scope

- Embedding identity: `512:1`
- Embedding representation: BF16 `[151552,4096]`
- Requested rows: `4`
- Unique rows: `4`
- Embedding source bytes: `32768`
- Embedding scale bytes: `0`
- Embedding overfetch: `0`
- Full embedding table converted to F32: `NO`
- Base layers: `0..45` (`46` layers)
- Layer `0`: dense
- Layers `1..45`: MoE
- Routed experts per token: Top-8 from `128`
- Batch and sequence: `1 x 4`
- Hidden size: `4096`
- Vocabulary: `151552`
- Logits shape: `[1,4,151552]`
- Final-position index: `3`

The output head is persisted identity `512:0`, distinct from the embedding,
and was consumed in `8192`-row BF16 chunks. The model payload and semantic
sidecar were mapped read-only; no external tokenizer, Hugging Face,
Safetensors, GGML, or device source was accessed.

## Production Evidence

| Item | Result |
|---|---:|
| Input hash | `1cc196e775e6023a3d9c83bb1a56865113fd64a6b387170cfdb8420860949d75` |
| Final transformer hash | `f1bbfbf9269833fc852bdf33d92ebbf8c2098011346a3455b5e42e1ee3c6fe13` |
| Final norm hash | `a9b25d1bce92ba5029b1248bf01fa2cdb6753d9a4983ad24db98a137a7e175e5` |
| Logits hash | `5c0c52e803670402726c729f53e81415efbb80113fc8d7748169c7f0d2d24afc` |
| Cumulative source bytes | `25247329280` |
| Requested source bytes | `25247329280` |
| Source overfetch | `0` |
| Unique model bytes | `25247329280` |
| Peak source range | `630094336` |
| Peak converted F32 cache | `2514575872` |
| Peak activations | `5980160` |
| Peak internal working set | `2520588800` |
| Peak KV state | `1507328` |
| Peak output-head chunk F32 | `134234112` |
| Output-head working set | `136790016` |
| Unselected routed tensors | `0` at every layer |
| Cross-layer KV reads | `0` |
| Active transient leases after run | `0` |
| Active execution/layer states after run | `0` |

The read-only mmap RSS after release was `24770372 KiB`; this is mapped source
page residency, not a model-sized F32 copy. The bounded runtime working-set
and converted-cache measurements above are reported separately.

## Independent Reference

The reference used the manifest’s validated persisted ranges and independently
computed attention, router stages, selected experts, shared experts, final
normalization, and logits. It did not use production intermediates for the
propagated replay.

- Embedding max absolute error: `0`
- Embedding max relative error: `0`
- Transformer max absolute error: `6.22558594e-03`
- Transformer max relative error: `1.36210525e+00`
- Final norm max absolute error: `1.41143799e-04`
- Final norm max relative error: `2.07366534e-02`
- Logits max absolute error: `6.10351562e-05`
- Logits max relative error: `1.06533945e+00`
- Production/reference final argmax: `220` / `220`
- Production/reference final top-5: `[220,198,25,32327,54026]`
- Production/reference final top-10: `[220,198,25,32327,54026,369,279,1437,271,697]`

The corrected real-input router diagnostic found two slots:

- Layer `24`, token `3`, rank `2`: production `54`, reference `104`.
- Layer `24`, token `3`, rank `3`: production `104`, reference `54`.

Both production/reference sets were identical. The K/K+1 cutoff margin was
`0.012577056884765625` (`6594` ULPs), and the common-input routed-MoE maximum
absolute delta was `4.76837158203125e-07`. This is a same-set ordered rank
instability caused by propagated F32/reduction-order drift, not a production
membership-selection defect. Production retains its specified rank-ordered
expert accumulation and lower-index tie-break behavior.

## Repeatability And Verification

The first full checkpoint and a full replay were byte-identical. After sorting
manifest expert records by `(expert, role)`, two independent depth-1 real-text
runs also produced byte-identical checkpoints and manifests. This fixes
qualification artifact nondeterminism without changing model execution.

Final Rust verification:

```text
cargo test -p vbuf-runtime -p vbuf-ml
```

Result: all tests passed, including the new generic embedding lookup tests.

Production command:

```text
cargo run --manifest-path rust/Cargo.toml -q -p vbuf-runtime --bin vbuf-runtime-step32h-real-text -- \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.semantic.vbuf \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32h-full.checkpoints \
  /tmp/opencode/step32h-full.manifest \
  Test
```

Independent reference command:

```text
python -u research/results/vbuf-ml-integration/step32g_bounded_reference.py \
  /tmp/opencode/step32h-full.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32h-full.checkpoints \
  46 real Test
```

Real-input route diagnostic:

```text
python -u research/results/vbuf-ml-integration/step32g_router_near_tie.py \
  /tmp/opencode/step32h-full.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32h-full.checkpoints \
  44 real
```

## Boundary

This qualifies real persisted-tokenizer input, bounded embedding-row
materialization, portable generic F32 execution of the persisted base stack,
selected-only MoE acquisition, chunked final projection, final-position logits
parity, and cleanup. It does not qualify decode, generation, device execution,
GGML parity, or strict full-chain exact ordered route parity. The model
artifacts and temporary checkpoint traces remain outside the repository.
