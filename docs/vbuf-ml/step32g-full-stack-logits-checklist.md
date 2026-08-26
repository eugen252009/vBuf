# Step 32G Full Stack And Logits Checklist

- [x] Execute persisted base layers `0..45` in order.
- [x] Keep the additional predictor-layer index space out of the base stack.
- [x] Use real activation handoff after the synthetic initial hidden state.
- [x] Resolve final norm and output head from persisted tensor identities.
- [x] Establish distinct embedding and output-head payload ranges.
- [x] Execute final RMSNorm through the generic portable graph.
- [x] Execute output projection through the generic portable MatMul graph.
- [x] Materialize output-head rows in bounded `8192`-row chunks.
- [x] Produce logits with shape `[1,4,151552]`.
- [x] Touch no unselected routed expert tensor in production.
- [x] Keep cross-layer KV reads at zero.
- [x] Release transient leases and execution state at teardown.
- [x] Run an independent persisted-range numerical reference.
- [x] Match final argmax and top-10 against the independent reference.
- [ ] Claim strict full-chain independent ordered route-ID equality; the two
  documented differences are rank swaps within one selected expert set and
  remain a numerical near-tie qualification note.
- [ ] Qualify tokenization, decode, generation, device execution, or GGML.

Evidence: `research/results/vbuf-ml-integration/step32g-full-stack-logits.md`.
