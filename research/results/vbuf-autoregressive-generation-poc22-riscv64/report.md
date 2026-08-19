# POC22 Autoregressive Native Generation

Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`

## Result

- Seed token: `0`.
- Generated positions: `1`.
- Reference sequence: `95626`.
- vBuf sequence: `95626`.
- Full sequence parity: PASS.
- Generated token feedback: PASS (each runtime output is the next embedding input).
- Runtime uses reference future: NO.
- Logits parity: PASS; max abs error `0`.
- Router selection parity: PASS.
- State alias violations: `0`.
- Final runtime state bytes: `552960`.
- Peak resident bytes: `66603008`.
- Peak active persistent bytes: `12607488`.
- Tracked source bytes/reload bytes: `608192512 / 0`.
- Residency hits/misses/evictions: `1532 / 1503 / 686`.
- Total elapsed time: `653990142846 ns`.
- Whole-model/whole-block/whole-packed-expert loading: `NO / NO / NO`.
- Execution-prep copied/repacked/transcoded bytes: `0 / 0 / 0`.

## Scope

This qualifies a bounded native autoregressive recurrence on x86. It does not claim tokenizer, sampler, chat, production, GPU, ARM32, RV2, or large-model completeness.
