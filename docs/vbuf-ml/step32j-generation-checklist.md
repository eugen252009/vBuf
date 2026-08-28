# Step 32J Repeated Autoregressive Generation Checklist

- [x] Use the persisted tokenizer and fixed real text `Test`.
- [x] Prefill token IDs `[51,68,82,83]` and retain checked KV state per layer.
- [x] Execute eight consecutive greedy decode steps through all 46 layers.
- [x] Reuse the request-local KV state without prefix recomputation.
- [x] Append exactly one KV position per layer per decode step.
- [x] Preserve ordered activation handoff across every layer and step.
- [x] Dispatch only selected Top-8 routed experts and the required shared expert.
- [x] Record bounded source, KV, converted-weight, activation, and timing telemetry.
- [x] Run the independent causal reference without production intermediates.
- [x] Match production/reference token IDs at all eight steps.
- [x] Match production/reference argmax and top-10 ordering at all eight steps.
- [x] Match routing membership and ordered expert decisions at all steps.
- [x] Verify zero unselected expert acquisition and zero expert overfetch.
- [x] Persist and recover reference progress atomically across interrupted runs.
- [x] Verify cleanup leaves zero active leases, execution states, layer states, and KV bytes.
- [x] Qualify the final replay with `REFERENCE_EXIT_STATUS=0` and `GENERATION_COMPLETE=YES`.
- [ ] Qualify device execution or GGML parity.

Evidence: `research/results/vbuf-ml-integration/step32j-repeated-autoregressive-generation.md`.
