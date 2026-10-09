# Numerical contract system

`policy-v2.json` is the current versioned registry for numerical, compatibility,
execution-invariant, and lifecycle contracts. `policy-v1.json` is retained as
historical policy input. The JSON Schema validates document structure;
`generate_numerical_contracts.py` performs stricter policy checks and emits the
C++ registry together with the SHA-256 of the exact policy bytes. Do not edit
generated files in the build tree.

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
  reference identity, exact candidate and qualification-run identities,
  model/backend/implementation, devices and SMs, placement, phase/topology,
  fixture/input/token identities, dtypes, shape, capacity, context, rows,
  measured metrics, criteria, invariants, and failure reasons. Scope fields
  omitted by a contract default to `*`; empirical limits belong in
  `empirical_scope`.
- Only a live, in-memory, active `PASS` evaluation under the current policy
  digest can be considered for admission. The result carries a private
  in-process integrity snapshot; mutation, default-constructed objects, and
  JSON/CSV replay cannot recreate that authority. Replay remains explicitly
  marked `replayed_metrics_only` and may reproduce historical classifications,
  but cannot authorize a candidate.
- **Trust boundary:** this is an integrity and admission mechanism, not a
  sandbox against hostile native code in the same process. The qualification
  runner and the provenance of the reference/candidate tensor views are trusted
  inputs. A caller able to execute arbitrary code in-process can call the live
  evaluator with fabricated tensor views; deployment must not treat this
  contract system as cryptographic attestation or as permission to load
  untrusted native capabilities.

## Current Qwen3 contracts

The admitted model is Qwen3-14B Q4_K_M, SHA-256
`f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`, using
GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` and the qualified 26/14
placement (RTX 3060 blocks 0–25; RTX 2080 SUPER blocks 26–39 plus final
norm/head).

- Final hidden/logit compatibility v2 retains the established
  `relative RMS <= 0.02 AND cosine >= 0.9998` semantics and now requires exact
  output vector shapes 5,120 and 151,936 respectively. V1 is retained for
  historical classification; ExecutionPlan candidates require v2 explicitly.
  The contracts are scoped to the model, CUDA placement, F32 outputs/F16 inputs,
  and declared capacity, context, and row bounds. These are not per-element or
  universal model tolerances.
- The direct synthetic AV operation contract compares native and packed AV
  independently to the FP64 oracle. The shared packed ceiling remains strict
  `max absolute error < 1e-3`; native AV preserves its stricter synthetic-oracle
  assertion `< 3e-6`. A separate contract measures native vs packed AV
  compatibility. These synthetic direct tests do not prove full-model
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

## ExecutionPlan admission and lifecycle audit

The lifecycle audit found that public candidate registration accepted caller-set
`Valid` state, executable guard definitions were not bound to their candidate
identity, qualification inputs did not carry the runtime plan/facts into the
validation transition, and separate output records could be combined without a
shared run identity. Active evaluation records also exposed mutable public
fields without an integrity check. The fixes below close those demonstrated
admission gaps; they do not make native in-process inputs cryptographically
trustworthy.

Admission is a `Candidate -> Valid` transition, not a caller-supplied status:

1. Registration accepts only `Candidate` state and clears caller-supplied
   evidence. Prebound dispatch definitions are optimizer-generated; native AV
   registration requires the canonical factory definition to match the supplied
   plan, including every guard and versioned requirement. Caller-selected
   executable guards cannot reuse a known candidate identity.
2. Validation binds evidence to the exact candidate ID, implementation identity,
   plan identity, runtime facts, active contract ID/version, policy digest,
   model/reference, device/SM/placement, dtypes, output shape, and a non-empty
   qualification-run ID. Each required contract must have exactly one record;
   both records must belong to the same run and agree on all shared provenance.
3. Only validated required records are retained. Hotness remains necessary.
   Historical replay, wrong-version, failed, mutated, duplicated, mixed-run,
   wrong-candidate, or out-of-scope evidence cannot authorize the transition.
4. Selection rechecks the stored evidence against the current candidate and
   runtime facts. Any mismatch selects canonical execution and reports a
   structured `EvidenceRejectionReason`; explicit invalidation clears evidence
   and keeps the cached candidate unselectable. A later matching runtime can
   still use a valid candidate after a scope-only fallback.
5. Unvalidated execution trials are compiled only when the explicit
   `VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS=ON` qualification option is
   set. The default is OFF; a trial is reported as selected but never as
   production-eligible.

The C++ admission tests include positive controls and adversarial cases:

| Case | Expected result |
|---|---|
| Exact canonical candidate descriptor, active v2 hidden/logit evidence, same run, matching runtime | `Candidate -> Valid`; ENABLED may select |
| Caller-supplied `Valid`, forged guards under a known identity, non-executable probe | Registration/transition rejected |
| Missing output, genuine numerical FAIL, wrong candidate or contract version | No transition |
| Mixed run, duplicate record, mutated or default-constructed PASS object | No transition |
| Mutated reference provenance, wrong implementation/runtime facts, or v2 output shape | Integrity/scope rejection; runtime mismatch selects canonical |
| Replay-only evidence or invalidated cached candidate | Rejected; canonical remains selected |
| Unvalidated qualification trial | Off by default; opt-in trial never reports production eligibility |

Native-layout AV remains experimental; it has
not been promoted. Its guards remain unchanged (capacity at most 512, 32-row
prefill or one-row decode through pre-decode context 32), and canonical packed-V
remains authoritative. The known prefill/topology and short-output failures
have not passed the active model-output contracts. Production capacity 1,032,
prefill chunk 32, single-GPU default, optimizer `SHADOW`, thresholds, and
canonical fallback are unchanged.

## Independent tensor-operation reference layer

`include/vbuf_high_precision_reference.h` and
`src/vbuf_high_precision_reference.cpp` provide a bounded, host-only independent
reference implementation for raw QK dot products, stable scaled/masked softmax,
GQA attention AV, last-axis RMSNorm, and general 2-D matmul. Views carry logical
shapes plus signed byte strides; outputs are owned, contiguous tensors and are
rounded explicitly to F32 or F64. Default output allocation is capped at 16M
elements. Matmul/QK/AV support explicit F32 or F64 left-to-right accumulation;
FP64 products and additions are separately rounded (contraction is prevented).
Softmax and RMSNorm use binary64. Qwen3 QK returns raw dot products and records
the `1/sqrt(head_dimension)` scale separately at softmax. GQA maps each query
head to `floor(query_head / (query_heads / kv_heads))`. Causal logical extents
and masks are explicit; excluded softmax positions are exactly zero and an
all-masked row is rejected.

`decode_ggml_rows` uses pinned GGML type traits for F16 and supported quantized
types (including Q4_K), plus the upstream Q8_K dequantizer where this GGML
revision exposes no public type-trait decoder. It verifies exact row byte
geometry and upstream row validation, and returns represented values as F32.
Matmul over those values measures arithmetic on the represented quantized
weights; it does not
attribute quantization error to the kernel. The GGML Q4_K test compares the
shared decoder output with the pinned `to_float` trait, routes a deterministic
F32-source/Q4_K reconstruction through the existing
`vbuf.quantization.weight_reconstruction_accuracy` contract, and exercises the
dequantized values in a binary64-accumulated matmul. That reconstruction
contract remains `NEEDS_CALIBRATION`; the synthetic source test does not qualify
Qwen quantization loss.

Reference evaluations continue to use the existing policy/contract identity.
Additional record fields bind each result to the reference implementation
revision, accumulation precision, input/output representation, and operation
parameters. These fields are part of the live evaluation integrity snapshot and
serialized JSON. Historical tensor captures are explicitly converted to
`replayed_metrics_only`; their result cannot authorize an ExecutionPlan. The
QK, softmax, RMSNorm, and general matmul contracts remain `NEEDS_CALIBRATION`;
no threshold, contract status, admission requirement, or production behavior was
changed. This provenance is qualification metadata, not cryptographic
attestation: the runner, input tensor provenance, and native process remain
inside the documented trust boundary.

The probability-row normalization diagnostic uses the explicit bound
`8 * F32 epsilon * sqrt(active_count)`; this is not a numerical-accuracy gate.

`tests/vbuf_high_precision_reference_contract.cpp` covers strided/transposed
views, F32-vs-F64 accumulation, GQA mapping, causal extents, masks and invalid
rows, RMSNorm epsilon, Q4_K decoding, output bounds, and contract integration.
`qualification/qwen3_reference_capture_qualification.cpp` recomputes QK,
softmax, and AV from the saved actual-model layer-0 prefill/decode captures,
checks the saved FP64 AV oracle bitwise, and emits replay-only policy-v2 JSON
records. Results and scope are documented in
[`qwen3-independent-tensor-reference-qualification.md`](../../../../research/results/vbuf-ml-integration/qwen3-independent-tensor-reference-qualification.md).

## Executable qualification

The C++ metric evaluator is in `src/vbuf_numerical_contracts.cpp`; direct
contract coverage is in `tests/vbuf_numerical_contracts_contract.cpp` and
ExecutionPlan admission coverage is in `tests/qwen3_execution_plan_contract.cpp`.
The native AV CPU/CUDA contract test evaluates its FP64-oracle and packed-parity
results through the registry. Qwen sequence, multi-GPU, and capacity
qualification paths emit machine-readable live contract results.

`tests/vbuf_numerical_contract_evidence_replay.cpp` consumes the preserved
historical topology CSV and actual-model AV-oracle metrics CSV. It checks the
known native prefill/topology logits failures and passing controls against the
current policy, and records actual-model AV metrics as `NOT_TESTED` under
`NEEDS_CALIBRATION`. All records remain marked replay-only. This is regression
classification, not a new model qualification claim.

`qualification/qwen3_numerical_calibration.py` orchestrates provenance-checked
Qwen3-14B CUDA capture replays. It verifies the saved Q/K inputs across the
capacity sweeps, runs the independent C++ reference evaluator, replays the
captured Q/K through MMVF/MMF/cuBLAS geometry on both CUDA devices, records the
selected GGML dispatch branch using temporary instrumentation in the generated
build copy, and restores that copy before exit. Layer-21 QK, causal softmax,
RMSNorm, and canonical AV captures are evaluated where inputs are available.
Raw inputs remain at their existing paths; the run manifest records their
SHA-256 values and the small generated QK output fixtures. The policy stays at
version 2 and every observation remains non-authorizing. See
[`qwen3-cuda-numerical-calibration.md`](../../../../research/results/vbuf-ml-integration/qwen3-cuda-numerical-calibration.md)
and its separate, inactive
[`contract proposal`](../../../../research/results/vbuf-ml-integration/qwen3-cuda-numerical-calibration/contracts-proposal-v1.json).
