# Numerical contract system

`policy-v1.json` is the versioned registry for numerical, compatibility,
execution-invariant, and lifecycle contracts. The adjacent JSON Schema validates
the document; `generate_numerical_contracts.py` performs stricter policy checks
and emits the C++ registry with the SHA-256 of the exact policy bytes. Do not
edit generated files in the build tree.

## Contract and evidence semantics

- Bump the policy version for incompatible policy meaning. Bump a contract's
  version when its reference, scope, metrics, invariant set, or criteria change.
  Preserve historical result records; never silently reinterpret their version.
- Keep measured metrics, hard invariants, and tolerance criteria separate.
  Multiple required criteria aggregate with logical AND. `less_than` remains
  strict; `less_equal` and `greater_equal` remain inclusive. Units are explicit:
  absolute, fraction, percent, and permille.
- Relative RMS is `||candidate-reference||2 / ||reference||2`; RMS error is the
  error-vector RMS; cosine is computed after scale normalization. Relative RMS
  and cosine are unavailable for a zero-norm operand rather than hidden behind
  an epsilon floor. Shape, count, dtype, finite values, and logical-input
  equivalence fail closed.
- Reference identity is normative. FP64 operation oracles measure arithmetic
  against a rounded high-precision reference, not exact real arithmetic.
  Canonical execution measures compatibility. Quantized-weight and
  unquantized-weight references are distinct categories and must not be
  conflated with kernel arithmetic accuracy.
- Evidence records include policy and contract versions, policy digest,
  reference identity, model/backend/implementation, devices and SMs, placement,
  phase/topology, fixture/input/token identities, dtypes, shape, capacity,
  context, rows, measured metrics, criteria, invariants, and failure reasons.
  Scope fields omitted from a v1 contract default to `*`; a contract must state
  its empirical boundary in `empirical_scope`.
- `ACTIVE` is required for authoritative qualification. `PROVISIONAL`,
  `NEEDS_CALIBRATION`, and `DEPRECATED` can retain observations but cannot
  authorize candidate selection. Missing references/metrics and mismatched
  scopes are `NOT_TESTED`, `NOT_APPLICABLE`, or `INVALID_EVALUATION`, never a
  pass.
- CSV/log metric replay is explicitly marked `replayed_metrics_only`. It can
  re-evaluate the current criteria and reproduce historical PASS/FAIL labels,
  but `numerical_evidence_matches_runtime()` rejects it as candidate evidence.
  Only live tensor evaluation under the current policy digest may authorize a
  candidate.

## Current Qwen3 contracts

The admitted model is Qwen3-14B Q4_K_M, SHA-256
`f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`, using
GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` and the qualified 26/14
placement (RTX 3060 blocks 0–25; RTX 2080 SUPER blocks 26–39 plus final
norm/head).

- The final hidden/logit canonical-compatibility gate retains the established
  `relative RMS <= 0.02 AND cosine >= 0.9998` semantics. It is scoped to that
  model, CUDA, placement, F32 outputs/F16 inputs, and the declared capacity,
  context, and row bounds. It is not a per-element tolerance or a universal
  model tolerance.
- The direct synthetic AV operation contract compares native and packed AV
  independently to the FP64 oracle. The shared packed ceiling remains strict
  `max absolute error < 1e-3`; native AV also preserves its historical stricter
  synthetic-oracle assertion `< 3e-6`. A separate contract measures native vs
  packed AV compatibility. These synthetic direct tests do not prove full-model
  compatibility.
- Captured actual-model AV-oracle observations are scoped to the exact artifact,
  layer-0 fixture, 26/14 placement, CUDA SM86, capacity 512, and visible context
  32/33. The contract is `NEEDS_CALIBRATION`: live evaluation records metrics,
  but no model-input-independent AV tolerance is approved. Historical CSV replay
  remains non-authorizing.
- QK, softmax, RMSNorm, general matmul, quantization, and lifecycle contracts
  without approved scoped criteria remain `NEEDS_CALIBRATION`. Lifecycle
  invariants are hard booleans (reset, atomic logical progress, canonical
  recovery), not floating-point tolerances.

## ExecutionPlan integration and AV status

A candidate with required numerical contract IDs cannot transition to `Valid`
without one live `PASS` record per required ID under the current policy digest,
active contract version, candidate implementation identity, and contract scope.
At selection time the model/backend/implementation/device/SM/placement/dtype,
phase, capacity, context, rows, and any explicitly constrained fixture/topology
must match again. Otherwise the optimizer falls back to canonical execution
with `NumericalQualificationMissing`. Existing hotness and execution guards
remain in force.

Native-layout AV remains experimental; its candidate state remains `Candidate`,
not `Valid`. Its existing
guards remain unchanged (capacity at most 512, 32-row prefill or one-row decode
through pre-decode context 32); canonical packed-V remains authoritative. The
known prefill/topology and short-output failures have not passed the active
model-output contracts. Operation-level FP64 accuracy cannot override model
compatibility. Production capacity 1,032, prefill chunk 32, single-GPU default,
optimizer `SHADOW`, thresholds, and canonical fallback are unchanged.

## Executable qualification

The C++ metric evaluator is in `src/vbuf_numerical_contracts.cpp`; direct
contract coverage is in `tests/vbuf_numerical_contracts_contract.cpp`. The
native AV CPU/CUDA contract test evaluates its FP64-oracle and packed-parity
results through the registry. Qwen sequence, multi-GPU, and capacity
qualification paths emit machine-readable live contract results.

`tests/vbuf_numerical_contract_evidence_replay.cpp` consumes the preserved
historical topology CSV and actual-model AV-oracle metrics CSV. It checks the
known native prefill/topology logits failures and passing controls against the
current policy, and records the actual-model AV metrics as `NOT_TESTED` under
`NEEDS_CALIBRATION`. All records remain marked replay-only. This is regression
classification, not a new model qualification claim.
