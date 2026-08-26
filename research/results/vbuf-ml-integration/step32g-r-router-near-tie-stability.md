# Step 32G-R: Router Near-Tie Stability

Date: 2026-08-26

## Result

`ROUTING_SET_EQUAL_ORDERED_SLOT_NEAR_TIE_WITH_NUMERICAL_DRIFT`

The two reported full-replay routing mismatches were reproduced from the same
persisted model artifact and the same synthetic input. They are two ordered
Top-8 slots in one invocation, not two different expert-set decisions:

- Layer `43`, token `2`, zero-based rank `4`: production `73`, reference `11`.
- Layer `43`, token `2`, zero-based rank `5`: production `11`, reference `73`.

The production and reference Top-8 expert sets are identical. There is no
production-only or reference-only expert, and therefore no Top-K membership
divergence. Across the full `46`-layer run this is `2` mismatching slots out
of `1440` ordered routing slots (`180` router invocations).

## Evidence

The diagnostic replay was bounded at depth `44`, which includes the divergent
layer and avoids an unnecessary post-divergence replay. It reported:

```text
KNOWN_ROUTING_DIVERGENCE_COUNT=2
TOTAL_ROUTER_INVOCATIONS=172
TOTAL_ROUTING_SLOTS=1376
ROUTING_MISMATCH_COUNT=2
```

The full-run denominator is `180` invocations and `1440` slots because layer
`0` is dense and layers `1..45` each have four token-level router decisions.

At layer `43`, token `2`, the ordered windows were:

```text
production: 80,111,125,124,73,11,50,71,36,31,77
reference:  80,111,125,124,11,73,50,71,36,31,77
```

The production corrected-score values for the swapped pair were:

```text
expert 73 = 15.015913009643555
expert 11 = 15.015912055969238
gap      = 0.00000095367431640625
```

The reference values reversed the order:

```text
expert 11 = 15.015918731689453
expert 73 = 15.015911102294922
```

This internal pair gap is approximately one production-F32 ULP. It is not a
K/K+1 membership tie: the production K/K+1 cutoff margin was
`0.0025911331176757812` (`2717` ULPs), and the reference margin was
`0.0025892257690429688` (`2715` ULPs).

The router-stage comparisons showed:

- Router input maximum absolute delta: `1.9073486328125e-04`.
- Raw router-score maximum absolute delta: `6.651878356933594e-05`.
- Corrected-score maximum absolute delta: `1.52587890625e-05`.
- The swapped experts differed by `2..8` corrected-score ULPs.
- A measured input perturbation of maximum magnitude
  `1.52587890625e-05` flipped the observed ordered decision.

The common-input cross-feed results were:

- Production math on production input: production order (`73,11`).
- Reference math on production input: reference order (`11,73`).
- Production math on reference input: reference order (`11,73`).
- Reference math on reference input: reference order (`11,73`).
- Float64 diagnostic math on production input: production order (`73,11`).

These controls show that both accumulated activation drift and numerical
reduction/computation order are relevant at this ordered near tie. They do not
show a persistent semantic disagreement about which experts belong in Top-K.

The selected-set-preserving rank swap changed the common-input routed-MoE
output by `1.220703125e-04`. The naturally propagated layer-43 routed-MoE
output and layer output deltas were `1.5869140625e-03` and
`1.1138916015625e-03`, respectively.

## Classification

Both reported positions are classified as **numerical ordered-routing
instability**:

- Type: propagated F32 numerical drift plus reduction/order sensitivity.
- Top-K membership: unchanged.
- Expert payload identity: unchanged and verified from the persisted artifact.
- Production semantic defect: not indicated.
- Production routing change: not made.

The production implementation retains its specified lower-index tie-break and
rank-ordered expert accumulation. The independent reference remains a
qualification oracle, not a reason to force route parity or change production
semantics.

## Qualification Boundary

The result supports full-stack numerical qualification with an explicit
ordered-route near-tie note. It does not support a claim of strict ordered
route-ID equality against this independently reduced F32 reference. Final
argmax and top-10 logits remain exact as recorded in
`step32g-full-stack-logits.md`.

Diagnostic command:

```text
python -u research/results/vbuf-ml-integration/step32g_router_near_tie.py \
  /tmp/opencode/step32g-r-full.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32g-r-full.checkpoints \
  44
```

The compact deterministic output from this replay is committed in
`step32g-r-router-near-tie-output.txt`. Large checkpoint traces and model
artifacts remain outside the repository.
