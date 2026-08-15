# CCC Research Conclusion

## Status

```text
CCC_DIRECTION_REJECTED
CCC_RESEARCH_REMAINS_CLOSED
```

This is the canonical conclusion for the completed Contextual Correction Code
(CCC) research program. Read this document before proposing CCC-related work.
It reconciles the historical evidence; it does not alter or delete any original
report. No CCC representation is selected for vBuf-ML, and no implementation is
authorized.

## Scope

The program tested CCC as a family of low-bit weight representations combining
some subset of:

1. a tensor-, row-, column-, row+column-, or lane-derived baseline;
2. a learned or constrained residual reconstruction alphabet;
3. packed low-bit correction codes;
4. optional exceptional residual handling; and
5. direct packed compute.

The numerical evidence is Q8_0-relative rather than BF16/F32 ground truth. It
covers representative Qwen3-32B attention-key tensors, broad Stage-1 tensor
sampling, deterministic fit/validation/test separation, canonical GGML Q/K/IQ
controls, Gaussian probes, and captured real transformer activations. Results
from different layers are not averaged.

CCC closure rejects the tested representation family as a promotion candidate.
It does **not** reject compact compute-facing representations, sub-byte storage,
lazy loading, range-planned payload access, zero-materialization consumption,
register-local reconstruction, or direct CPU/GPU consumption in general.

## Research lineage

Repository state recorded before this closure:

```text
repository root: /home/eugen/projekte/vBuf
branch:          vbuf-ml
HEAD:            59673a1a5baddded8d74c91a71676d0b220c714a
working tree:    clean
```

Four lineages developed from the common pre-research commit `33a4d03` and were
preserved as independent Git parents before reconciliation:

| Lineage | Preserved evidence | Reconciliation |
|---|---|---|
| Canonical Stage-1/Stage-2 | `b9043fb`, `b9316d9` | foundation for later reconciliations |
| Parallel C3 hard gate/native kernel/fairness audit | `889bd58` | no-ff merge `d11d167`; reconciliation `1ea629b` |
| Structured-baseline matrix | `28588a0` | no-ff merge `53afb10`; reconciliation `df00863` |
| Geometric/C4 qualification | `da93e43` | no-ff merge `5131144`; reconciliation `59673a1` |

The graph, merge parents, and common merge base are part of the provenance: no
lineage was squashed, rebased, or rewritten to manufacture agreement.

Primary evidence:

- Stage-1: [`../../benchmark-results/vbuf-ml-step30-ccc-assessment/representation-assessment.md`](../../benchmark-results/vbuf-ml-step30-ccc-assessment/representation-assessment.md)
- Stage-2: [`../../benchmark-results/vbuf-ml-step31-ccc-canonical/stage2-report.md`](../../benchmark-results/vbuf-ml-step31-ccc-canonical/stage2-report.md)
- C3 hard gate: [`../../benchmark-results/vbuf-ml-step31-ccc-hard-gate/hard-gate-report.md`](../../benchmark-results/vbuf-ml-step31-ccc-hard-gate/hard-gate-report.md)
- Native C3: [`../../benchmark-results/ccc-c3-native-kernel/native-kernel-report.md`](../../benchmark-results/ccc-c3-native-kernel/native-kernel-report.md)
- Runtime fairness audit: [`../../benchmark-results/ccc-c3-native-fairness-audit/fairness-audit-report.md`](../../benchmark-results/ccc-c3-native-fairness-audit/fairness-audit-report.md)
- Structured baseline: [`../../benchmark-results/ccc-structured-baseline-qualification/ccc_structured_baseline_qualification.md`](../../benchmark-results/ccc-structured-baseline-qualification/ccc_structured_baseline_qualification.md)
- Geometric qualification: [`../../benchmark-results/ccc-geometric-qualification/ccc_geometric_qualification.md`](../../benchmark-results/ccc-geometric-qualification/ccc_geometric_qualification.md)
- C4 hard gate: [`../../benchmark-results/ccc-c4-hard-gate/c4-hard-gate.md`](../../benchmark-results/ccc-c4-hard-gate/c4-hard-gate.md)
- Reconciliations: [C3/orientation](../../benchmark-results/vbuf-ml-step32-ccc-reconciliation/reconciliation-report.md), [structured baseline](../../benchmark-results/vbuf-ml-step33-ccc-structured-reconciliation/reconciliation-report.md), and [C4](../../benchmark-results/vbuf-ml-step34-ccc-c4-reconciliation/reconciliation-report.md)

## Corrected historical chronology

### Stage-1

**HISTORICAL RESULT:** Learned nonlinear scalar levels appeared superior to
simplified uniform/affine controls at 2–6 bpw. C2/C3 therefore appeared
interesting before canonical IQ controls and real hidden states were available.
Zero and tensor-mean anchors tied, row/column statistics generally hurt, and
lane-mod32 helped only C2 modestly.

**SUPERSEDED INTERPRETATION:** The apparent gain established a nonlinear scalar
quantization signal, not a contextual-prediction advantage and not
competitiveness against canonical formats.

### Stage-2

**HISTORICAL RESULT:** Pinned canonical controls and real layer-32 hidden states
showed learned C3/C4 dominated, preserved C2 only as an extreme-rate niche, and
left learned C4 plus fixed 0.1% FP16 residual exceptions numerically Pareto.
Power levels approximated learned levels. The report classified the bounded
program as `CCC_LOW_BIT_NICHE_INTERESTING` at that time.

**SUPERSEDED INTERPRETATION:** Later independent evidence removed the remaining
C3 promotion rationale, rejected the structured premise, and showed that
geometry itself provides no advantage over a strong free control. The original
measurements and contemporary interpretation remain historical evidence.

### Parallel C3 lineage

**HISTORICAL RESULT:** Its hard gate placed C3 between Q2_K and Q3_K and
recommended a native kernel. Its packed scalar, BMI2, and AVX2 implementations
then proved direct three-bit W*x and initially appeared substantially faster
than canonical formats.

**SUPERSEDED INTERPRETATION:** Reconciliation found that the hard-gate evaluator
mapped flat GGML storage with `reshape(5120,1024).T` instead of the correct
`reshape(1024,5120)`. Serialized quantized bytes and dequantized tensors were
identical; a common eight-vector test localized the discrepancy to orientation.
Corrected Stage-2 evidence, including IQ2_XS, makes C3 numerically dominated.
The initial speed comparison also timed C3 fused packed compute against
canonical dequantization plus dense FP32 dot, not canonical native dot kernels.
The fairness audit rejects the speed advantage. Its prose and structured/raw
latencies disagree in exact magnitude, so no exact ratio is promoted; their
canonical-faster direction agrees.

**FINAL STATUS:** C3 is numerically dominated and has no accepted runtime
advantage. Direct packed compute feasibility remains valid as a separate
systems result.

### Structured-baseline lineage

**HISTORICAL RESULT:** Row+column separation reconstructed the intended algebra
to approximately `1.4e-15` relative error.

**FINAL STATUS:** The algebra is valid, but tensor, row, column, and combined
predictors left residual RMS/raw at approximately 1.0 and supplied no useful
signal. Algebraic validity is not empirical utility.

### Geometric/C4 lineage

**HISTORICAL RESULT:** An early bounded search made geometric C4 appear slightly
better than its then-current free-codebook control while canonical controls
were blocked.

**SUPERSEDED INTERPRETATION:** A strengthened deterministic Lloyd-Max control
won FIT, validation, untouched TEST, and real hidden-state W*x. Canonical
Q3/Q4/IQ controls then dominated both plain free and geometric C4.

**FINAL STATUS:** Free C4 dominates geometric C4; plain C4 is canonically
dominated.

## Hypothesis decomposition and final classifications

### H1 — Contextual / structured baseline

Tensor means are effectively tied with zero. Row, column, row+column, and lane
metadata are neutral or harmful after held-out testing and byte accounting.
The structured matrix reports residual RMS/raw of approximately 1.0.

```text
STRUCTURED_BASELINE_REJECTED
```

### H2 — CCC separation algebra

The row+column decomposition invariant holds to approximately `1.4e-15`
relative error.

```text
SEPARATION_ALGEBRA_VALID
```

Mathematically valid does not imply empirically useful.

### H3 — Power-geometric reconstruction alphabet

Power geometry can approximate some learned level sets. It does not beat a
sufficiently optimized free scalar codebook at equal storage: strengthened
Lloyd-Max wins FIT, validation, untouched TEST, and real hidden-state W*x.

```text
GEOMETRIC_ALPHABET_NOT_COMPETITIVE
```

`GEOMETRY_APPROXIMATES_LEARNED` remains a narrow descriptive observation, not
an advantage or promotion classification.

### H4 — Exact zero state at very low bit width

At C3, exact-zero or duplicate-zero layouts generally spend the eight-state
budget less effectively than no-zero/mid-riser layouts. This is a reusable
low-bit scalar-quantization observation, not evidence for contextual CCC.

```text
NO_ZERO_LOW_BIT_OBSERVATION_SUPPORTED
```

### H5 — C3 rate/quality

Corrected real-hidden-state evidence includes:

| Candidate | True bpw | Mean relative W*x error |
|---|---:|---:|
| IQ2_XS, activation-aware | 2.3125 | 0.1148 |
| Learned C3 | about 3.0001 | 0.2016 |

IQ2_XS is both smaller and more accurate, while nearby canonical Q3/IQ3
controls are also stronger. Runtime is excluded from this classification.

```text
C3_NUMERICALLY_DOMINATED
```

### H6 — C3 direct packed compute

The native implementation consumes exact three-bit packed codes, including
cross-byte extraction, through scalar packed, BMI2, and AVX2 W*x paths without
constructing a complete dense weight tensor.

```text
C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED
SUB_BYTE_DIRECT_COMPUTE_FEASIBLE
```

This physical feasibility does not make C3 a competitive quantizer.

### H7 — C3 runtime superiority

The original comparison was unfair. Once canonical native GGML dot kernels
replace per-row dequantization plus dense FP32 dot, the claimed C3 advantage
disappears and canonical kernels are faster. Exact speed ratios are not
promoted because fairness-audit narrative and structured/raw latency values
conflict.

```text
C3_RUNTIME_ADVANTAGE_REJECTED
```

### H8 — Plain C4

Layer-0 real-hidden-state evidence includes:

| Candidate | True bpw | Mean relative W*x error |
|---|---:|---:|
| Free C4 | about 4.0064 | 0.1530 |
| Geometric C4 | about 4.0063 | 0.1997 |
| IQ3_XXS | 3.0625 | 0.0620 |
| Q3_K | 3.4375 | 0.0412 |

The same free-over-geometric and canonical-over-plain ranking appears in the
independent layer-32 evidence. Values across layers are not averaged.

```text
C4_FREE_DOMINATES_GEOMETRIC
C4_CANONICALLY_DOMINATED
```

### H9 — Sparse C4 residual tail

Stage-2 learned C4 plus fixed 0.1% sparse FP16 residual exceptions reported
approximately `4.0481 bpw` and `0.0556` real W*x error and remained numerically
Pareto in that experiment. The exact representation was not tested by the later
C4 lineage; direct apply, sparse layout, and runtime cost remain unresolved.

```text
C4_SPARSE_TAIL_SURVIVES_UNREPLICATED
EVIDENCE_ONLY
```

This is not CCC success and is not implementation authorization. The candidate
no longer materially depends on the defining contextual or geometric CCC
premises.

## Final evidence table

| Direction | Final status | Evidence basis |
|---|---|---|
| Contextual baseline | REJECTED | Stage-1 plus structured-baseline lineage |
| Separation algebra | VALID | approximately `1.4e-15` invariant error |
| Power geometry | NOT COMPETITIVE | strengthened free controls and C4 hard gate |
| Exact-zero C3 state | NO-ZERO OBSERVATION SUPPORTED | state-budget ablations |
| C3 numerical | DOMINATED | Stage-2 canonical/IQ controls |
| C3 direct apply | FEASIBLE | native packed scalar/BMI2/AVX2 paths |
| C3 runtime advantage | REJECTED | native-kernel fairness audit |
| Plain C4 | DOMINATED | C4 hard gate and reconciliation |
| Sparse C4 tail | UNREPLICATED; EVIDENCE ONLY | Stage-2 only |
| CCC overall | REJECTED | combined reconciled evidence |
| Sub-byte direct compute | FEASIBLE | native C3 systems experiment |

## Why CCC is rejected

The defining CCC premises did not survive stronger controls:

- contextual predictors contribute no useful held-out residual signal;
- constrained geometry does not beat a strong free codebook;
- C3 is numerically dominated;
- plain C4 is numerically dominated; and
- the claimed C3 runtime advantage fails an equivalent native-kernel control.

The isolated sparse-tail observation does not rescue the family because it is
unreplicated and no longer depends materially on contextual prediction, power
geometry, or C3.

```text
CCC_DIRECTION_REJECTED
```

## Surviving findings independent of CCC

### Sub-byte direct compute is feasible

The native C3 experiment proves this systems chain:

```text
packed stored representation
    -> load packed bytes
    -> register-local bit extraction / reconstruction
    -> multiply or FMA
    -> accumulate
```

No whole-tensor FP16/FP32 materialization is required. Whether the encoded
quantizer is numerically good is a separate question.

```text
SUB_BYTE_DIRECT_COMPUTE_FEASIBLE
```

### Zero-copy does not require immutable bit representation

For this research boundary, zero-copy/zero-materialization means that stored
bytes are already a valid runtime representation and do not require creation
of another complete tensor before compute. Register-local interpretation is
compatible with that principle:

```text
storage representation -> direct consumption -> register-local decode -> compute
```

By contrast, this is the materialization path to avoid:

```text
storage representation -> full reconstructed tensor -> compute
```

This is a research principle, not a wire-format or production terminology
change.

### Real hidden states are a mandatory promotion gate

Gaussian W*x probes were useful for screening but often differed strongly from
real transformer activations. In the C4 hard gate, canonical formats became
far better on real inputs while scalar/geometric candidates became worse.
Future ML representation promotion therefore requires untouched captured real
activations.

```text
REAL_ACTIVATION_GATE_REQUIRED
```

### Strong controls are mandatory

A constrained alphabet must face a sufficiently optimized unconstrained scalar
codebook at equal storage. A custom runtime kernel must face native canonical
kernels performing equivalent work, not dequantization/reference paths.

```text
STRONG_FREE_CONTROL_REQUIRED
NATIVE_RUNTIME_CONTROL_REQUIRED
```

### Tensor orientation must be explicit and testable

An apparent `(1024,5120)` matrix can still have the wrong flat-storage mapping.
For these GGML tensors the required invariant is:

```python
W = flat.reshape(1024, 5120)  # W[out, input]
y = X @ W.T
```

Future harnesses should, where practical, preserve the source FP32 hash,
serialized quantized hash, dequantized hash, and a small common-vector output
hash. Shape alone is insufficient.

```text
EXPLICIT_TENSOR_ORIENTATION_REQUIRED
```

## Sparse-tail evidence boundary

The surviving observation may be reconsidered only as a new, independently
scoped hypothesis:

```text
low-bit base quantization + sparse residual exceptions
```

It must not inherit CCC naming, contextual baselines, power geometry, C3
assumptions, or authorization from this program. Any work requires a separate
proposal covering independent replication, true storage, sparse index/layout,
direct apply, and fair runtime. This document does not create that proposal.

## Reopening criteria

A new model, tensor, predictor variant, optimizer tweak, gamma value, or broad
parameter sweep is not reopening evidence. CCC may be reconsidered only if new
pre-existing evidence materially changes a rejected premise, for example:

- a predictor substantially reduces held-out residual energy on multiple real
  model tensors;
- a constrained CCC representation beats a strong equal-storage free codebook
  on untouched real-activation tests;
- a CCC representation is non-dominated against strong canonical controls in
  rate × functional quality; or
- a clearly defined systems advantage survives an equivalent native-runtime
  comparison and is large enough to justify inferior numerical efficiency.

Without such evidence:

```text
CCC_RESEARCH_REMAINS_CLOSED
```

Speculative qualification or parameter search is not authorized as a way to
create reopening evidence.

## vBuf-ML boundary after closure

CCC closure rejects only the tested CCC representation family. The clean future
systems boundary remains:

> Which numerically strong low-bit representations can vBuf-ML store and consume
> with minimal materialization and transport overhead?

This document deliberately does not answer that question and authorizes no
representation, kernel, quantizer, GPU, sparse-tail, wire-format, or production
work.

## Final classifications

```text
CCC_DIRECTION_REJECTED

STRUCTURED_BASELINE_REJECTED

SEPARATION_ALGEBRA_VALID

GEOMETRIC_ALPHABET_NOT_COMPETITIVE

C3_NUMERICALLY_DOMINATED

C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED

C3_RUNTIME_ADVANTAGE_REJECTED

C4_FREE_DOMINATES_GEOMETRIC

C4_CANONICALLY_DOMINATED

C4_SPARSE_TAIL_SURVIVES_UNREPLICATED

SUB_BYTE_DIRECT_COMPUTE_FEASIBLE

REAL_ACTIVATION_GATE_REQUIRED

CCC_RESEARCH_REMAINS_CLOSED
```

## Provenance authority

Historical reports remain authoritative for what each experiment measured and
believed at its commit. Reconciliation reports are authoritative for identified
defects and cross-lineage interpretation. This document is authoritative for
the final CCC decision and future reopening boundary.
