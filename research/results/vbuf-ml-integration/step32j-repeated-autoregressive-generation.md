# Step 32J: Repeated Autoregressive Generation

Date: 2026-08-28

## Result

`REPEATED_AUTOREGRESSIVE_GENERATION_QUALIFIED`

The portable generic Rust runtime executed the fixed real-text input `Test`,
retained one checked KV state per layer, and performed eight consecutive greedy
decode steps through all 46 GLM-4.5-Air-FP8 layers. The independent persisted-
range NumPy reference reproduced the same causal sequence without consuming
production intermediates.

Production and reference selected the same token at every step, matched the
final top-10 ordering at every step, and recorded zero routed-expert membership
or order mismatches. Cleanup released all transient leases, execution states,
layer states, and logical KV bytes.

## Qualified Input And Sequence

- Text: `Test`
- Prefill token IDs: `[51,68,82,83]`
- Layers: `0..45` (`46`)
- Layer `0`: dense
- Layers `1..45`: MoE
- Routed experts: Top-8 from `128`
- Vocabulary: `151552`
- Maximum new tokens: `8`
- Production/reference generated IDs: `[220,16,25,220,16,13,576,2629]`
- Generated text: `Test 1: 1. The sum`
- Generated token-ID SHA-256: `5562b8d3244c7dfa39855762637b3586c7b41d46c14ef33fd4b881c8e4639439`
- Generated text SHA-256: `8cf8c65881d0708c861426d44d398882864949592ef2232a1d682374e5e5840f`
- Termination: `MaxNewTokens`

The production path used the persisted tokenizer, semantic sidecar, and
validated payload ranges. It did not access Hugging Face, Safetensors, GGML,
remote sources, or a loader-specific source callback.

## Production Evidence

| Item | Result |
|---|---:|
| Prefill token IDs | `[51,68,82,83]` |
| Prefill KV state bytes | `1507328` |
| Prefill source bytes | `1241522176` |
| Prefill time | `70062.521 ms` |
| Decode steps | `8` |
| KV positions appended per step | `46` |
| KV bytes appended per step | `376832` |
| Decode source bytes per step | `13463473152` |
| Peak converted model weight working set | `1061249536` |
| Peak activation bytes | `806912` |
| Peak total working set | `1062154752` |
| Unselected expert tensors touched | `0` |
| Expert overfetch bytes | `0` |
| Prefix recomputation | `NO` |
| Historical QKV recomputation | `NO` |
| Full token activation history retained | `NO` |
| Cleanup | `PASS` |
| Active transient leases after cleanup | `0` |
| Active execution states after cleanup | `0` |
| Active layer states after cleanup | `0` |
| Logical KV state bytes before/after cleanup | `4521984` / `0` |

Per-step production results were:

| Step | Consumed | Next | Past reads | Logits max abs | Decode ms | Tokens/sec |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 220 | 16 | 184 | `4.76837158e-05` | `879786.285` | `1.13663968` |
| 2 | 16 | 25 | 230 | `5.53131104e-05` | `890367.409` | `1.12313186` |
| 3 | 25 | 220 | 276 | `5.91278076e-05` | `893960.926` | `1.11861712` |
| 4 | 220 | 16 | 322 | `5.81741333e-05` | `889196.431` | `1.12461090` |
| 5 | 16 | 13 | 368 | `5.19752502e-05` | `895579.318` | `1.11659568` |
| 6 | 13 | 576 | 414 | `4.14848328e-05` | `895499.594` | `1.11669509` |
| 7 | 576 | 2629 | 460 | `3.65674496e-05` | `869755.894` | `1.14974788` |
| 8 | 2629 | 315 | 506 | `3.62396240e-05` | `869828.398` | `1.14965205` |

Every decode step made `45` routing decisions and selected `360` expert
occurrences. State length progressed from `4` after prefill to `12` after the
eighth decode step. The production completion record reports
`GENERATION_COMPLETE=YES`, `GENERATION_CLEANUP=PASS`, and
`HARNESS_EXIT_STATUS=0`.

## Independent Reference

The reference replay independently consumed the validated manifest ranges and
computed the causal prefix, attention, router stages, selected experts, shared
experts, final normalization, and logits. The final run was resumed from the
durable state after earlier host reboots and completed with exit status `0`.

| Item | Result |
|---|---:|
| Generated token sequence parity | `True` |
| Routing membership mismatches | `0` |
| Routing order mismatches | `0` |
| Argmax mismatches | `0` |
| Top-10 mismatches | `0` |
| Maximum transformer absolute difference | `3.96728516e-04` |
| Maximum transformer relative difference | `2.375` |
| Maximum per-step logits absolute difference | `5.91278076e-05` |
| Maximum per-step logits relative difference | `1.11796904` |
| Reference exit status | `0` |

Per-step logits maximum absolute differences were
`[4.76837158e-05, 5.53131104e-05, 5.91278076e-05, 5.81741333e-05,
5.19752502e-05, 4.14848328e-05, 3.65674496e-05, 3.62396240e-05]`.
The qualification claims bounded numerical parity, token parity, argmax/top-10
ordering parity, and routing parity; it does not claim byte identity of
independently computed floating-point activations.

## Recovery And Host Stability

Earlier reference attempts were interrupted by host reboots while the machine
was running a graphical workload. The append-only current log preserves those
partial traces, while `reference.current-completed-at`, `reference.state.json`,
and the final step summaries identify the successful run. The reference state
file provided atomic step recovery; the payload page-release change kept the
observed reference RSS below approximately `1.3 GiB` with more than `49 GiB`
available memory during the successful final-step replay.

The final replay was performed after moving the second GPU to a dedicated PSU
PCIe cable and closing the GPU-heavy game workload. The host remained up past
the previous failure point and through reference completion. This is stability
evidence for the qualification environment, not a causal proof about the
earlier resets. Device execution and GGML parity remain outside this step.

## Evidence And Verification

Production evidence:

- `.step32c/step32j-generation/gate-8/run.log`
- `.step32c/step32j-generation/gate-8/generation.checkpoints.progress`

Reference evidence:

- `.step32c/step32j-generation/gate-8/reference.current.log`
- `.step32c/step32j-generation/gate-8/reference.current-completed-at`
- `.step32c/step32j-generation/gate-8/reference.state.json`

Qualification commands:

```text
cargo test --workspace
cargo fmt --all -- --check
python -m py_compile research/results/vbuf-ml-integration/step32g_bounded_reference.py
bash -n research/results/vbuf-ml-integration/run-step32j-generation.sh
bash -n research/results/vbuf-ml-integration/run-step32j-reference.sh
```

The immutable model artifacts and large checkpoint traces remain outside the
repository. No generic vBuf wire format, runtime ownership boundary, device
path, GGML integration, or llama.cpp loader behavior was changed.

## Boundary

This qualifies repeated real-text autoregressive generation with retained KV,
ordered 46-layer execution, selected-only MoE acquisition, independent causal
reference parity, bounded working-set telemetry, and cleanup. It does not
qualify device execution, GGML parity, or backend-loader integration.
