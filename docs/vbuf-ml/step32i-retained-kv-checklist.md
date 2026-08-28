# Step 32I Retained-KV One-Token Decode Checklist

- [x] Use the persisted tokenizer and fixed real text `Test`.
- [x] Prefill token IDs `[51,68,82,83]` and decode token `220`.
- [x] Retain one checked KV state for every layer after prefill.
- [x] Decode with query length `1`, past length `4`, position `4`, and capacity `5`.
- [x] Reuse the same request-local state objects for prefill and decode.
- [x] Read exactly four prior KV positions and append exactly one per layer.
- [x] Execute all 46 layers with ordered activation handoff.
- [x] Dispatch only the selected Top-8 experts in MoE layers.
- [x] Record and compare per-layer KV state snapshots.
- [x] Run the independent five-token causal reference.
- [x] Match routing membership, final argmax, and final top-10 ordering.
- [x] Verify no prefix recomputation, token 6 execution, or generation loop.
- [x] Verify cleanup leaves zero active leases, execution states, and layer states.
- [x] Run `cargo test --workspace` and the Python syntax check.
- [x] Qualify repeated generation through the Step 32J gate.
- [ ] Qualify device execution or GGML parity.

Evidence: `research/results/vbuf-ml-integration/step32i-retained-kv-one-token-decode.md`.
