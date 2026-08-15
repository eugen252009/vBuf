# Step 30 addendum — structured algorithm matrix

Status: **PARTIAL — 22/22 tensor families tested; hidden-state and deeper group/layer phases remain**.

The previous completion claim and `B` classification are invalid. This
addendum records the corrected tensor/group/layer search boundary. The
full applicability ledger is in:

```text
benchmark-results/vbuf-ml-step30-reparameterization/algorithm-applicability.csv
```

Every one of the 22 requested algorithm families is represented at every one
of the seven boundaries. Unimplemented families are explicitly marked `NOT
TESTED` with a reason; no family is silently dropped.

## ALGORITHM APPLICABILITY MATRIX

The machine-readable matrix covers:

```text
Tensor, QKV, Attention, Gate/Up, MLP, Whole Layer, Cross Layer
```

The complete family set is:

```text
Low Rank
Sparse
Low Rank + Sparse
Kronecker
Tensor Train
Butterfly
Generalized Butterfly
Toeplitz
Circulant
Low Displacement Rank
Codebook
Vector Quantization
Block Affine
Sign / Magnitude
Exponent / Mantissa
Generator Function
Fourier Basis
Hadamard
Structured Orthogonal
Polynomial / Matrix Function
Block Dictionary
Exact / Numerically Equivalent Reparameterization
```

## TENSOR-LEVEL WINNERS

A real 0.6B layer-0 `attn_k` tensor was screened with all 22 required families.
Each `TESTED` row has deterministic 64-byte-aligned experimental serialization
accounting, random-probe action error, direct compact apply timing, temporary
memory, bytes touched, and writer fit time.

The former five gaps now have bounded MPO, Butterfly, generalized Butterfly,
genuine nilpotent-shift displacement-rank, and Householder implementations.
All performed poorly on random action probes. Details are in
`step30-continuation.md`. No winner is promoted because real hidden-state
validation is absent.

## QKV JOINT SEARCH

Q, K, and V consume the same input and were safely stacked along their output
partition using the 0.6B layer-0 tensors.

| Representation | Independent raw factor bytes | Joint raw factor bytes | Raw-factor reduction | Action error |
|---|---:|---:|---:|---:|
| rank-16 low rank | 458,752 | 327,680 | 1.40x | 0.955 |

The joint factor preserves Q/K/V output partitions. The result is lossy and
not runtime-qualified.

## GATE / UP JOINT SEARCH

Gate and up consume the same input and were jointly stacked.

| Representation | Independent raw factor bytes | Joint raw factor bytes | Raw-factor reduction | Action error |
|---|---:|---:|---:|---:|
| rank-16 low rank | 524,288 | 458,752 | 1.14x | 0.975 |

The down projection was not concatenated because it consumes the post-gating
activation rather than the same input.

## ATTENTION GROUP SEARCH

QKV joint low rank was tested. Complete attention-group search was not tested:
the output projection has a different input domain and no complete attention
functional evaluator was added.

## MLP GROUP SEARCH

Gate/up joint low rank was tested. Complete MLP search was not tested because
down projection consumes a different intermediate domain.

## WHOLE-LAYER ALGORITHMS TESTED

No complete-layer algorithm was claimed as tested. The applicability matrix
contains an explicit `NOT TESTED` entry for every whole-layer family because a
complete nonlinear layer evaluator and layer-level compact execution object
were not implemented in this bounded step.

This is intentional: QKV and gate/up were not incorrectly promoted to a
whole-layer result.

## WHOLE-LAYER PURE RESULTS

Not tested. The canonical layer useful-byte and Step-29 resident-compute
values remain the measurement authority.

## WHOLE-LAYER RESIDUAL DIAGNOSTICS

Not tested because residuals require a fitted whole-layer candidate. Tensor and
group residual metrics are present in the candidate and joint result files.

## WHOLE-LAYER HYBRIDS

Not tested. No pure whole-layer survivor justified hybrid search.

## SHARED PARAMETERS FOUND

At the tested QKV and gate/up boundary, joint low-rank factors shared a single
stacked factorization. The joint factor was counted once, with output rows
partitioned back into role-specific operators.

## SHARED INPUT TRANSFORMS FOUND

The joint low-rank form performs one shared `Vx` operation followed by one
shared stacked `U` projection. Role-specific output partitions are recovered
without reconstructing dense matrices.

## STORAGE SHARING GAIN

Measured at rank 16:

```text
QKV:     1.40x raw-factor byte reduction
Gate/Up: 1.14x raw-factor byte reduction
```

These are explicitly **factor-array estimates**, not canonical serialized
storage gains and not gains over canonical Q8. Experimental metadata, padding,
and partition descriptors are reported separately. Both candidates have poor
action preservation.

## COMPUTE SHARING GAIN

The research harness now measures dense apply, one shared `Vx`, role-specific
`Uz`, total joint compact apply, independent compact apply, temporary bytes,
and representation/input/output bytes touched. These NumPy measurements are
0.6B random-probe controls, not llama runtime or hidden-state results.

## BEST INDEPENDENT REPRESENTATION

No inference-qualified independent winner exists. Scalar codebook and
block-exponent candidates entered residual diagnostics, while low-rank and
other highly compact structures had poor action preservation. None has real
hidden-state evidence.

## BEST GROUPED REPRESENTATION

The rank-16 QKV pilot had a 1.40x **raw-factor byte reduction** relative to
independent rank-16 factor arrays. This is not a serialized storage-gain claim,
and its approximately 0.955 random-probe action error prevents promotion.

## BEST WHOLE-LAYER REPRESENTATION

None. Whole-layer qualification is incomplete and remains explicitly
`NOT TESTED`.

## LEVEL COMPARISON

```text
canonical Q8 layer:
  qualified authority

independent tensors:
  tested for selected tensors; compact low-rank candidates were lossy

shared-input groups:
  QKV and gate/up tested; modest factor sharing, poor action preservation

attention + MLP groups:
  complete groups not tested across incompatible domains

whole layer:
  not tested; no complete nonlinear evaluator
```

## TRUE LAYER STORED INFORMATION

No compact whole-layer representation was emitted. Consequently no shared,
private, residual, or metadata bytes are claimed for a whole-layer candidate.
The group result files count factor bytes once and include all factor arrays.

## LAYER EFFECTIVE BITS PER WEIGHT

Not applicable to a qualified whole-layer candidate. Canonical Step-29 layer
accounting remains approximately 518,104,064 useful bytes for the measured
32B layer context.

## LAYER DIRECT COMPUTE

Not measured for a compact whole-layer object. Group factor application is
measurable in the joint result records, but it is not a llama runtime result.

## LAYER FUNCTIONAL ERROR

Not measured. Complete-layer validation must preserve RMSNorm, RoPE, attention,
softmax, causal behavior, gated activation, residual additions, and KV state;
none of those semantics were replaced here.

## LAYER MEMORY-WALL EFFECT

The Step-29 host-local simulation context remains:

```text
canonical useful layer bytes: approximately 518,104,064
resident layer compute:       approximately 16.15 ms median
```

Correct decimal-GB/s canonical controls are recorded for all five scenarios:

```text
1.4 GB/s: 370.074 ms
3.2 GB/s: 161.908 ms
8 GB/s:    64.763 ms
16 GB/s:   32.382 ms
32 GB/s:   16.191 ms
```

They use only the Step-29 32B context and are not combined with 0.6B compact
pilots. These are simulation controls, not overlap claims.

## CROSS-LAYER RESULT

Not tested. The conditional adjacent-layer phase was not justified because no
whole-layer winner exists.

## COMPLEMENTARY ALGORITHMS

No residual-driven complementary hybrid was selected. The family matrix
retains Low Rank + Sparse and all other families for future screening rather
than mislabeling unimplemented combinations as negative experimental results.

## BEST LAYER HYBRID

None. No whole-layer hybrid was emitted.

## JOINT-REPRESENTATION CLASSIFICATION

```text
F — inconclusive
```

The group result is specifically bounded: QKV and gate/up share-input low-rank
factors, but their functional errors are too high for runtime integration.

## Completion limits

```text
canonical artifacts changed: 0
wire changes: 0
BaseShift changes: 0
runtime changes: 0
production compact representation emitted: 0
cross-layer search: not tested
whole-layer direct evaluator: not implemented
```
