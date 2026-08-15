# Step 30 addendum — structured algorithm matrix

This addendum records the corrected tensor/group/layer search boundary. The
full applicability ledger is in:

```text
benchmark-results/vbuf-ml-step30-reparameterization/algorithm-applicability.csv
```

Every one of the 21 requested algorithm families is represented at every one
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
```

## TENSOR-LEVEL WINNERS

The prior tensor-level qualification remains the only tensor-level numerical
winner search. Low-rank factors achieved very low storage but unacceptable
action error at the tested ranks. The remaining families are retained in the
matrix and were not silently treated as low-rank substitutes.

## QKV JOINT SEARCH

Q, K, and V consume the same input and were safely stacked along their output
partition using the 0.6B layer-0 tensors.

| Representation | Independent bytes | Joint bytes | Storage gain | Action error |
|---|---:|---:|---:|---:|
| rank-16 low rank | 458,752 | 327,680 | 1.40x | 0.955 |

The joint factor preserves Q/K/V output partitions. The result is lossy and
not runtime-qualified.

## GATE / UP JOINT SEARCH

Gate and up consume the same input and were jointly stacked.

| Representation | Independent bytes | Joint bytes | Storage gain | Action error |
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
QKV:     1.40x versus independent factors
Gate/Up: 1.14x versus independent factors
```

These are storage gains over the tested low-rank candidates, not gains over
canonical Q8 storage. Both candidates have poor action preservation.

## COMPUTE SHARING GAIN

Shared compute was structurally identified as one common input projection, but
no production runtime was changed and no authoritative end-to-end compute
speedup was claimed.

## BEST INDEPENDENT REPRESENTATION

Within the prior bounded tensor search, plain rank-16 low rank was the simplest
compact direct representation. It remained highly lossy.

## BEST GROUPED REPRESENTATION

Rank-16 joint QKV low rank had the best measured group storage gain: 1.40x
relative to independent rank-16 factors. Its action error was approximately
0.955 on random probes.

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

Storage times were recorded for 1.4, 3.2, 8, 16, and 32 GB/s in
`whole-layer-results.csv`. These are simulation controls, not overlap claims.

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
B — meaningful group-level structure, but not a qualified whole-layer result
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
