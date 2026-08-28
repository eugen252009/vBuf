# Step 32H Real Text To Full-Stack Logits Checklist

- [x] Use the persisted tokenizer metadata and runtime tokenizer path.
- [x] Encode the fixed UTF-8 qualification text `Test` without added specials.
- [x] Validate the persisted tokenizer round trip and token IDs `[51,68,82,83]`.
- [x] Gather embedding rows by validated token ID through the generic runtime.
- [x] Read only four embedding rows and avoid full-table F32 materialization.
- [x] Execute the real embedding through layers `0..45` with ordered activation handoff.
- [x] Execute final RMSNorm and the persisted LM head through generic graphs.
- [x] Keep LM-head materialization bounded to `8192` vocabulary rows per chunk.
- [x] Produce final logits with shape `[1,4,151552]` and qualify the last position.
- [x] Run the independent bounded numerical reference from the recorded input.
- [x] Match embedding parity and final-position argmax/top-10.
- [x] Verify zero unselected routed-expert acquisition and zero cross-layer KV reads.
- [x] Verify transient cache, lease, execution-state, and layer-state cleanup.
- [x] Verify deterministic real-input checkpoints and deterministic manifest emission.
- [ ] Claim strict full-chain independent ordered route-ID equality; two same-set
  rank swaps remain a documented near-tie exception.
- [x] Qualify retained-KV decode and repeated generation through the Step 32I/J
  gates.
- [ ] Qualify device execution or GGML parity.

Evidence: `research/results/vbuf-ml-integration/step32h-real-text-full-stack.md`.
