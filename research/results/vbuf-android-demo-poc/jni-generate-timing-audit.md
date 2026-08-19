# JNI Generate Timing Audit

Status: historical diagnostic measurement. The full four-token call did not
reach a terminal result within the bounded observation windows. No runtime or
model semantics were changed.

This report's approximately `109.7 s` position is `HISTORICAL_DIAGNOSTIC`
evidence from the earlier qualification-heavy path. It is not the controlled
runtime-mode A/B baseline. The later controlled measurements are recorded in
`normal-inference-baseline.md`:

```text
CONTROLLED_QUALIFICATION_POSITION_MS: 73588
CONTROLLED_NORMAL_POSITION_0_MS: 35397
CONTROLLED_NORMAL_POSITION_1_MS: 40060
CONTROLLED_POSITION_0_SPEEDUP: 2.079x
```

## Scope

- Device: Pixel 7 Pro, arm64-v8a Android build.
- Model: DeepSeek-V2-Lite IQ2_XXS.
- Residency budget: 256 MiB.
- Prompt: `Explain the purpose of bounded generation`.
- Requested generation limit: 4 tokens.
- Invocation: existing `NativeInference.open()` and `generate()` JNI path,
  triggered through a temporary diagnostic receiver.

## Call Path

`JNI.generate()` calls `DirectSession::generate()`. The method tokenizes the
prompt, runs one `run_step()` for every prompt token, then one `run_step()` per
generated token. Each step performs embedding, all 27 model layers, the output
head, and greedy selection. Each layer executes both the actual path and the
reference/control path used by the current parity implementation.

## Measurements

The run reached `OPEN_OK` and emitted continuous layer progress. Position 0
completed normally:

| Phase | Time |
| --- | ---: |
| Embedding | 32 ms |
| 27-layer sequence | 105,733 ms |
| Output head | 3,920 ms |
| Total step | 109,695 ms |
| Position total | 109,701 ms |

Position 1 also progressed continuously. It reached layer 14 during the final
30-second observation window. Representative layer totals ranged from about
1.99 s at layer 0 to 4.88 s at layer 9 and 3.78 s at layer 14. The per-layer
breakdown showed FFN work dominating most layers; the reference FFN path was
also material and is included in the sequence timing.

## Interpretation

The prior apparent ~32-minute stall is not explained by a single silent wait.
The direct path is performing repeated full model steps. One prompt position
already costs approximately 110 seconds on this device, before subsequent
prompt positions and four generated positions. This is sufficient evidence
that prompt prefill and repeated actual/reference execution dominate the
observed wall time. The controlled mode comparison subsequently separated the
reference path and established a faster normal-inference position baseline.

The measurements do not establish a complete `GENERATE total_ms`, prompt token
count, generated-token latency, or terminal parity result because the bounded
run was stopped before completion. They also do not separate materialization
wait time from compute within each layer; the layer intervals include both.

## Qualification Boundary

This audit is observational only. It does not claim generation success, router
parity, payload parity, or a transport regression. The temporary timing hooks
and diagnostic receiver were removed after capture. The existing qualification
controls and the historical router-parity failure remain unchanged.
