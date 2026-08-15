# Step 30 — reconstructable weight reparameterization qualification

Status: **complete as a bounded research qualification; no artifact or runtime
change**.

Step 30 tests whether useful mathematical action can be represented with fewer
stored degrees of freedom than the qualified Q8_0 tensors, without preserving
literal scalar values one-for-one.

## Scope

Selected real Qwen3 tensors:

```text
Qwen3-0.6B:
  blk.0.attn_output.weight
  blk.0.ffn_down.weight

Qwen3-32B:
  blk.0.attn_output.weight
```

The immutable Q8_0 GGUF tensors were used as the numerical oracle. The
corresponding qualified vBuf artifacts were hash-verified. No candidate was
serialized as vBuf and no canonical artifact was modified.

## REPARAMETERIZATION SEARCH

Tested representations:

```text
q8-reference
plain low-rank SVD factors, ranks 8/16/32
row-centered low-rank factors
row/column diagonal scaling plus a low-rank core
column permutation plus a low-rank core
```

The tests intentionally include parameter relocation into row means, diagonal
scales, and permutations. Butterfly was not blindly forced into this bounded
qualification; it was not tested because the direct low-rank/scaling candidates
were the selected compact direct-application comparators.

## INFORMATION RELOCATION

Information was relocated into:

```text
row means
row/column scale vectors
column permutations
low-rank factor matrices
```

All auxiliary information is included in stored-parameter accounting.

## TRUE STORED DEGREES OF FREEDOM

For each candidate the evidence records:

```text
stored scalar parameters
stored bytes
auxiliary parameters
rank
matrix dimensions
```

The Q8_0 reference counts 32 int8 values plus one fp16 scale per 32-weight
block. Low-rank candidates count every factor and auxiliary vector/permutation.

## EFFECTIVE BITS PER ORIGINAL WEIGHT

Representative 32B attention-output results:

| Candidate | Stored bits/weight | Matrix error | Action error |
|---|---:|---:|---:|
| Q8 reference | 8.500 | 0 | 0 |
| SVD rank 8 | 0.081 | 0.981 | 0.985 |
| SVD rank 16 | 0.163 | 0.973 | 0.977 |
| SVD rank 32 | 0.325 | 0.961 | 0.964 |
| row-centered rank 16 | 0.166 | 0.972 | 0.967 |
| diagonal-scaled rank 16 | 0.173 | 0.976 | 0.978 |
| permuted rank 16 | 0.169 | 0.973 | 0.970 |

The tested ranks are highly compact but do not preserve useful action at these
settings. Lower storage alone is not sufficient.

## RECONSTRUCTION CLASS

```text
q8-reference: EXACT qualified original representation
all tested compact candidates: LOSSY
```

No exact cross-tensor algebraic reparameterization was claimed. The tested
scaling, centering, and permutation transforms are mathematically defined, but
the low-rank truncation makes their final candidates lossy.

## DIRECT FUNCTION PRESERVATION

Validation included:

```text
matrix reconstruction error
random action error for W*x
maximum random action error
```

For the 32B selected tensor, rank-16 plain SVD had approximately 97.7% relative
action error on the random probes. Row-centering and permutation were slightly
better in some metrics but remained approximately 97% error.

Real hidden-state validation was unavailable because no suitable existing
consumer hook was exposed for capturing Qwen hidden-state distributions in
this bounded step. Random action probes are therefore the reported Level-2
validation, not a claim of downstream model equivalence.

## WRITER-SIDE COST

Randomized SVD fitting was measured separately and treated as offline-only.
Representative 32B rank-16 fit costs were hundreds of milliseconds for the
selected tensor in this environment. Runtime direct application uses only the
compact factors and does not reconstruct a full dense matrix.

## REPRESENTATION DISCOVERY

Plain low-rank factors were the simplest direct-application representation.
Row/column scaling and column permutations did not materially rescue action
quality at the tested ranks. Butterfly was not shown to win or lose; it was
outside this bounded comparator set.

The result does not justify writing a new compact tensor format or changing the
runtime. It does justify retaining reparameterization as a future research
axis, but only with stronger action-level and downstream validation.

## Constraints

```text
wire changes: 0
BaseShift changes: 0
Nano changes: 0
Nested-vBuf changes: 0
payload repack: 0
quantization changes: 0
canonical artifact changes: 0
runtime dense copy required by candidate: 0
```

Evidence:

```text
benchmark-results/vbuf-ml-step30-reparameterization/
```
