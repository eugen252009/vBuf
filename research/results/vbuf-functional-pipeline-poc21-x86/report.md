# POC21 Functional Pipeline

Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`

## Acceptance Results

- Real token embedding rows for IDs 0 and 1: exact parity after native dequantization.
- Transformer body: `blk.0..blk.26`, both positions, exact reference parity.
- Activation handoff: zero-copy, `activation_boundary_copy_bytes=0`.
- Final RMSNorm and quantized output projection: exact logits parity.
- Vocabulary: `102400` logits.
- Greedy next token: token 0 -> `86711`; token 1 -> `86711`.
- Per-layer runtime state isolation: PASS.
- Unselected expert graph/tensor/source activity: zero.
- Model artifact mutation: NO.
- Execution-prep copy/repack/transcode bytes: zero.

## Residency

- Policy: `COST_AWARE`.
- Capacity: `268435456` bytes.
- Peak resident: `268434432` bytes.
- Logical persistent bytes: `4777135104`.
- Peak active persistent bytes: `12607488`.
- Loads: `3061`; hits: `6143`; misses: `5991`.
- Evictions: `2895`; reload events: `2264`; reload bytes: `1835014464`.
- Policy decisions: `2895`; candidate evaluations: `540244`.

The full raw qualification output is captured during execution in `/tmp/poc21-end-to-end-v3.log`.
