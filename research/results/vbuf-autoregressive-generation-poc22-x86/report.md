# POC22 Autoregressive Native Generation

Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`

## Result

`NATIVE_INTERNAL_RECURRENCE: PASS` for the bounded x86 trace. Overall POC22
remains `BLOCKED` for the requested milestone until the pinned llama.cpp/GGUF
autoregressive oracle and recurrence-specific failure cases are executed.

- Seed token: `0`.
- Generated positions: `4`.
- Reference sequence: `86711, 86711, 86711, 86711`.
- vBuf sequence: `86711, 86711, 86711, 86711`.
- Full sequence parity: PASS.
- Generated token feedback: PASS (each runtime output is the next embedding input).
- Runtime uses reference future: NO.
- Logits parity against the independent native reference graph: PASS; max abs error `0`.
- Router selection parity: PASS.
- State alias violations: `0`.
- Embedding runtime identity collisions: `0` (repeated token `86711` reuses its own identity; it does not alias another row).
- Materializer identity collisions: `0`.
- Final runtime state bytes: `2211840`.
- Peak resident bytes: `268120064`.
- Peak active persistent bytes: `12607488`.
- Tracked source bytes/reload bytes: `2337161216 / 1662449984`.
- Residency hits/misses/evictions: `6399 / 5749 / 2929`.
- Total elapsed time: `1921722296487 ns`.
- Whole-model/whole-block/whole-packed-expert loading: `NO / NO / NO`.
- Execution-prep copied/repacked/transcoded bytes: `0 / 0 / 0`.
- Final normalization and LM-head paths are reused at every position; direct
  per-position final-norm instrumentation is not separate from logits parity.

The pinned llama.cpp/GGUF preparation is now reproducible from two fresh
checkouts; see the `llama-*` evidence files. The actual pinned DeepSeek GGUF
oracle reaches two positions but produces `[19304, 19304]`, diverging at the
first position from the native reference's `[86711, 86711]`. The four-position
oracle run was stopped at first divergence, so this remains an external-oracle
semantic blocker rather than a preparation blocker.

## Per-Position Results

| Position | Input token | Reference next token | vBuf next token | Logits max abs | State bytes | Tracked source bytes | Reload bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 86711 | 86711 | 0 | 552960 | 608192512 | 0 |
| 1 | 86711 | 86711 | 86711 | 0 | 1105920 | 608192512 | 573193216 |
| 2 | 86711 | 86711 | 86711 | 0 | 1658880 | 608192512 | 581439136 |
| 3 | 86711 | 86711 | 86711 | 0 | 2211840 | 512583680 | 507817632 |

## State, Memory, and I/O

- Runtime state added per token: `552960` bytes for this four-position trace.
- Peak active persistent bytes: `12607488` at every position.
- Peak resident bytes by position: `268120064`, `267624448`, `267128832`, `267128832`.
- Tracked source bytes: `2337161216`; tracked reload bytes: `1662449984`.
- Activation and backend-workspace bytes were not instrumented separately.
- Average, median, P95, and P99 active bytes were not instrumented separately.
- Per-position timings are in [`timing-by-token.json`](timing-by-token.json).

## Evidence Files

- [`audit.md`](audit.md)
- [`generation-fixture.json`](generation-fixture.json)
- [`reference-sequence.json`](reference-sequence.json)
- [`runtime-sequence.json`](runtime-sequence.json)
- [`position-parity.json`](position-parity.json)
- [`router-parity.json`](router-parity.json)
- [`state-growth.json`](state-growth.json)
- [`memory-by-token.json`](memory-by-token.json)
- [`source-by-token.json`](source-by-token.json)
- [`recurrence-trace.json`](recurrence-trace.json)
- [`failure-results.md`](failure-results.md)

## Scope

This qualifies a bounded native autoregressive recurrence against the independent
native reference graph on x86. It does not claim pinned llama/GGUF autoregressive
parity, tokenizer, sampler, chat, production, performance, GPU, ARM32, RV2, or
large-model completeness. Recurrence-specific failure injection and a repeated
determinism run were not part of the final repository qualification gates.
