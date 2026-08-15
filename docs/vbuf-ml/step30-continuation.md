# Step 30 continuation — missing families and residual quantization

## STEP

Step 30 continuation. Step 31 has not started.

## STATUS

**PARTIAL — 22/22 tensor families tested; hidden-state and deeper group/layer phases remain.**

## MISSING-FAMILY COMPLETION

All five former gaps now have real 0.6B layer-0 `attn_k` runs, serialized
accounting, action error, and direct apply:

| Family | Capacities | Best observed action error | Result |
|---|---|---:|---|
| Tensor Train / MPO | ranks 2/4/8, 4^5 tensorization | 0.997 | highly compact, very slow contraction |
| Butterfly | one 10-stage sparse branch | 0.999 | failed action preservation |
| Generalized Butterfly | two distinct permuted branches | 1.000 | no rescue from extra branch |
| Low Displacement Rank | nilpotent shift, ranks 2/4, two signs | >3.5 | displacement approximation amplified error |
| Structured Orthogonal | 4/8 Householders + two diagonals | 0.991 | failed action preservation |

Synthetic direct-apply tests cover identity MPO, identity Butterfly, LDR
reconstruction/apply agreement, and Householder norm preservation.

## 22-FAMILY BREADTH STATUS

```text
tested:     22 / 22
not tested:  0 / 22
```

This completes tensor breadth only, not Step 30 as a whole.

## HIERARCHICAL 4+4 RESULTS

| Variant | True bits/weight | Action error |
|---|---:|---:|
| linear coarse + linear residual | 8.003 | 0.0351 |
| learned primary + global residual | 8.003 | 0.0471 |
| learned primary + conditional residual | 8.011 | **0.0184** |
| best block-local error, block 32 | 40.002 | 0.00385 |
| best compact block-local, block 512 | 10.002 | 0.0152 |

Block-local tables improve error by relocating substantial information into
per-block tables; they are not 8-bit representations after true accounting.

## F32 VS F64 FITTING

F64 fitting changed hierarchical and sign/residual errors only negligibly.
It increased writer time in some runs and did not change persistent storage.
No F64 runtime recommendation follows.

## SIGN + MAGNITUDE + RESIDUAL RESULTS

| Allocation | True bits/weight | Action error |
|---|---:|---:|
| 1+4+3 | 8.007 | 0.0193 |
| 1+3+4 | 8.007 | **0.0189** |

Both use learned magnitude levels and conditional residual tables. Sign bits,
indexes, tables, metadata, and padding are counted.

## VECTOR PROTOTYPE + RESIDUAL RESULTS

Best observed by width:

| Width | Variant | True bits/weight | Action error |
|---:|---|---:|---:|
| 2 | conditional/output | 4.019 | 0.116 |
| 4 | conditional/output | 2.036 | 0.345 |
| 8 | conditional/output | 1.069 | 0.593 |
| 16 | conditional/output | 0.635 | 0.756 |

Output-oriented grouping generally won. The Python qualification path
materializes a decoded dense tensor for vector apply; this is explicitly not
the intended fused runtime path, so its timing is diagnostic only.

## CONDITIONAL VECTOR RESIDUAL RESULTS

Conditional dictionaries improved width-2/4/8 action relative to global
residual dictionaries, at the cost of larger dictionaries. Dictionary overhead
is included in the effective bits/weight.

## CODEBOOK / INDEX ENTROPY

Primary and residual empirical entropies are reported per candidate. Typical
hierarchical correction entropy was close to four bits, so the nominal nibbles
did not expose a large obvious entropy-coding opportunity. No variable-length
coding was implemented.

## BYTE-BUDGET COMPARISON

`byte-budget-comparison-v2.csv` compares old and new candidates around 0.5, 1,
2, 4, 8, 16, and 32-bit regions using true experimental serialized bytes.

## EXISTING BASELINE COMPARISON

At approximately eight bits, hierarchical conditional and sign/magnitude
residual candidates materially improve over the prior flat scalar codebook
(approximately 0.018–0.019 versus 0.150 action error), but provide only about
1.06x reduction from Q8_0. Near one bit, vector residual improves on prior VQ
(approximately 0.593 versus 0.704) but remains far from useful preservation.

## NEW TENSOR PARETO FRONTIER

The three-axis frontier is recorded in `tensor-pareto-v2.csv`. Extreme global
structures remain compact but inaccurate. Residual quantization shifts the
error frontier substantially at 4–8 bits, while current Python decode cost is
high.

## RESIDUAL DIAGNOSTICS

Hierarchical diagnostics include near-zero mass, top-magnitude concentration,
entropy, spectral concentration, conditional index entropy, and code
utilization. Vector diagnostics include primary/residual entropy and dead
codewords.

## COMPUTE-FOR-BANDWIDTH RESULT

The best-error new candidates reduce Q8 bytes only approximately 1.06x while
incurring large Python decode inflation. Lower-bit vectors reduce bandwidth
2–16x but retain unacceptable action error. No production trade is qualified.

## DIRECT APPLY RESULT

MPO, Butterfly, generalized Butterfly, LDR, structured orthogonal,
hierarchical scalar, and sign/residual candidates apply compact parameters
directly without reconstructing full W. Vector candidates retain the explicit
Python dense-decode limitation. Native fused nibble/vector kernels were not
implemented.

## MEMORY-WALL SCALE PROJECTION

`memory-wall-projection-v2.csv` is labeled:

```text
SCALE PROJECTION — NOT MEASURED 32B COMPRESSION
```

It projects only measured 0.6B tensor compression ratios onto the 518,104,064
byte Step-29 context at 1.4/3.2/8/16/32 GB/s. It makes no 32B functional claim.

## PROMOTED GROUP CANDIDATES

**NONE — tensor evidence insufficient and real hidden-state validation absent.**

## QKV FOLLOW-UP

Not performed.

## GATE / UP FOLLOW-UP

Not performed.

## TRUE STORED INFORMATION

Every new row separately reports indexes, primary tables, residual tables,
scales, signs, factors, permutations, payload, metadata, padding, partition
descriptors, and total serialized bytes.

## EFFECTIVE BITS PER WEIGHT

Best relevant regions span approximately 0.5–8.01 bits/weight. Low nominal
index width is never reported without dictionary overhead.

## ACTION PRESERVATION

Best new result near Q8 storage: conditional hierarchical 4+4, approximately
0.0184 random-probe action error. Best near one bit: conditional output-oriented
vector width 8, approximately 0.593 error. Neither has hidden-state evidence.

## COMPUTE INFLATION

Python hierarchical/sign decoding is hundreds of times slower than the fastest
NumPy dense timing in some samples. Vector qualification additionally decodes
dense W. These timings establish implementation cost, not a native fused-kernel
forecast.

## BEST NEW REPRESENTATION

Conditional hierarchical 4+4 gives the strongest action preservation at
approximately 8.01 bits/weight, but only modestly reduces Q8 storage.

## BEST NEW HYBRID

Sign + 3-bit learned magnitude + 4-bit conditional residual is competitive at
approximately 8.007 bits/weight and 0.0189 action error. It does not dominate
conditional hierarchical 4+4 on all axes.

## CLASSIFICATION

**F — inconclusive.** Tensor breadth is complete, but no low-bit candidate has
both useful action preservation and qualified direct runtime cost.

## WHOLE-LAYER GATE

**NOT YET.**

## VBUF WIRE FORMAT CHANGES

0

## VBUF-ML CANONICAL CHANGES

0

## BASESHIFT CHANGES

0

## NANO CHANGES

0

## RUNTIME ARCHITECTURE CHANGES

0

## SOURCE ARTIFACT CHANGES

0

## BLOCKERS / LIMITATIONS

```text
real hidden-state evaluation unavailable
vector direct apply uses temporary dense decode
no native fused nibble/vector kernel
single 0.6B tensor only
no 32B functional transfer evidence
no promoted group candidate
```

## ATTRIBUTION CONCLUSION

F64 offline fitting had negligible functional effect. Nominal codes differ
from true serialized widths when dictionaries are local or conditional.
Correction stages materially improve scalar action at roughly eight bits.
Vector sharing improves the 1–4 bit frontier but remains lossy. Python decode
cost dominates current direct-apply timing. The 32B results are storage-only
scale projections.

## NEXT RECOMMENDED STEP

Do not create a production representation. Capture real hidden states for the
same 0.6B tensor and compare conditional hierarchical 4+4 against Q8_0 and the
flat scalar codebook before any group or whole-layer promotion.
