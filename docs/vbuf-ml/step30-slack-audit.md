# Step 30 alignment-slack audit

## AUDIT

Existing vBuf-ML alignment slack and fixed primary/correction budgets. Generic
vBuf padding semantics were not changed.

## STATUS

**PARTIAL — physical slack audit complete; expanded vector/adaptive budget
addendum remains incomplete.**

## ARTIFACT PROVENANCE

The qualified 0.6B and 32B GGUF/vBuf hashes were revalidated. Exact planner
sizes equal the physical vBuf file sizes.

## GENERIC VBUF CHANGES

0

## VBUF-ML CANONICAL CHANGES

0

## BASESHIFT CHANGES

0

## TOTAL EXISTING SLACK — 0.6B

```text
43 bytes
0.0000067406% of artifact
0.0000005771 bits/weight
```

## TOTAL EXISTING SLACK — 32B

```text
44 bytes
0.0000001264% of artifact
0.0000000107 bits/weight
```

## SLACK DISTRIBUTION

| Artifact | Median | p95 | Max | zero blocks | 1–3 | 4–7 | >=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6B | 0 | 0 | 7 | 328 | 2 | 7 | 0 |
| 32B | 0 | 0 | 7 | 725 | 1 | 8 | 0 |

## SLACK BY TENSOR ROLE

Every model tensor role has zero alignment slack at selected BaseStep 8. All
43/44 bytes occur after small control/tokenizer payloads.

## SLACK BY LAYER

Every 0.6B and every 32B transformer layer has zero tensor slack. The 32B audit
covers all 64 layers plus global tensors.

## USABLE SLACK

0 bytes.

Canonical v0.6 padding must remain zero. Writing representation-local data into
it would make generic validation reject the artifact.

## STRANDED SLACK

```text
0.6B: 43 bytes
32B:  44 bytes
```

All physical slack is both too fragmented (maximum seven bytes) and prohibited
for nonzero canonical content.

## BASESHIFT SENSITIVITY

Selected BaseShift 3 is the only free baseline. Larger BaseSteps create paid
storage overhead:

| BaseStep | 0.6B extra bytes | 32B extra bytes |
|---:|---:|---:|
| 16 | 1,264 | 2,416 |
| 32 | 7,040 | 14,528 |
| 64 | 18,496 | 38,720 |
| 128 | 41,472 | 87,168 |
| 256 | 87,936 | 184,320 |

None is classified as free capacity.

## CORRECTION STRATEGIES TESTED

Sparse exceptions, packed corrections, tiny codebooks, block residuals, and
vector refinement were checked against the selected tensor's local budget.
Its actual budget is zero, so no correction object fits.

## CONDITIONAL 4+4 + SLACK RESULT

```text
before action error: 0.0183773
after action error:  0.0183773
used slack:          0 bytes
artifact growth:     0 bytes
```

## SIGN + MAGNITUDE + RESIDUAL + SLACK RESULT

No change; zero local slack.

## VECTOR WIDTH 2 + SLACK RESULT

No change; zero local slack.

## VECTOR WIDTH 4 + SLACK RESULT

No change; zero local slack.

## VECTOR WIDTH 8 + SLACK RESULT

No change; zero local slack.

## ERROR REDUCTION PER FREE BYTE

0. No free correction byte can be validly consumed.

## BEST USE OF EXISTING SLACK

Leave it canonical zero padding.

## PHYSICAL ARTIFACT SIZE BEFORE

```text
0.6B: 637,925,504 bytes
32B:  34,816,197,376 bytes
```

## PHYSICAL ARTIFACT SIZE AFTER

Identical, because no correction was inserted.

A same-length research mutation can put nonzero bytes in padding, but generic
behavior changes from accept to reject; it therefore fails the core invariant.

## OFFSET / EXTENT STABILITY

PASS for the unmodified artifacts. The rejected same-size mutation retains
offsets but fails generic-behavior stability.

## RUNTIME ACCESS COST

Zero measured correction cost because no valid correction exists. No future
fused-kernel gain is claimed.

## 32B LAYER-LEVEL OPPORTUNITY

All 64 layers have zero tensor slack. The theoretical free correction budget
per layer is zero bytes.

## THEORETICAL FREE CORRECTION BUDGET

43 bytes for the entire 0.6B artifact and 44 bytes for the entire 32B artifact,
all in control regions rather than tensor extents.

## PRACTICALLY USABLE CORRECTION BUDGET

0 bytes.

## FUNCTIONAL VALUE

**NEGLIGIBLE.** No selected model tensor has local slack.

## DOES SLACK CHANGE THE STEP-30 PARETO FRONTIER?

NO.

## ARCHITECTURAL VALUE

The concept is elegant, but canonical zero-padding semantics and measured
quantities eliminate practical value for these artifacts.

## PRIMARY / CORRECTION BUDGET SWEEP

A separate paid-byte scalar conditional-codebook sweep tested all requested
4-, 6-, and 8-bit splits. It does not use alignment slack.

## BEST 4-BIT TOTAL ALLOCATION

```text
2 primary + 2 correction
true width: 4.00195 bits/weight
action error: 0.12805
```

## BEST 6-BIT TOTAL ALLOCATION

```text
3 primary + 3 correction
true width: 6.00342 bits/weight
action error: 0.04384
```

## BEST 8-BIT TOTAL ALLOCATION

```text
3 primary + 5 correction
true width: 8.00928 bits/weight
action error: 0.01805
```

## PRIMARY-HEAVY VS CORRECTION-HEAVY

Balanced won at four and six bits. At eight bits, correction-heavy 3+5 narrowly
outperformed 4+4 on random-probe action.

## OPTIMAL OBSERVED SPLIT

3+5 at the eight-bit operating point.

## MARGINAL PRIMARY-BIT VALUE

Positive until the 3+5/4+4 neighborhood, then additional primary bits reduced
correction capacity and worsened action error.

## MARGINAL CORRECTION-BIT VALUE

At eight bits, moving from 4+4 to 3+5 reduced action error by approximately
0.00108. Further movement to 2+6 worsened it.

## FIXED-PHYSICAL-EXTENT RESULT

The best scalar splits above respect their modeled serialized extent budgets.
These are paid representation extents, not padding reuse.

## PRIMARY SHRINK / CORRECTION GROWTH RESULT

YES within the paid eight-bit budget: 3+5 beat 4+4. Shrinking the primary did
not create alignment slack; it reallocated paid representation bits.

## BLOCK-ADAPTIVE RESULT

Not tested. Zero local slack and absent real hidden states did not justify a
new adaptive representation search.

## VBUF SLACK AUGMENTATION RESULT

No augmentation: zero usable bytes.

## BEST SLACK HELPER

None.

## PHYSICAL SIZE GROWTH

```text
valid zero-cost correction: 0 bytes used, 0 growth
nonzero padding mutation:   0 growth, but invalid generic behavior
larger BaseStep:            actual positive growth
```

## DOES SMALLER PRIMARY + LARGER CORRECTION WIN?

**DEPENDS.** It wins at 8 bits (3+5 versus 4+4), but balanced allocations win
at 4 and 6 bits.

## RATE-DISTORTION CONCLUSION

Correction-heavy allocation can improve a fixed paid budget, but existing vBuf
alignment slack contributes no usable capacity.

## BLOCKERS / LIMITATIONS

```text
canonical padding must be zero
maximum slack fragment is seven bytes
all tensor extents have zero slack
vector split sweep not repeated
block-adaptive allocation not tested
real hidden-state objective unavailable
```

## ATTRIBUTION CONCLUSION

The planner imposes only 43/44 padding bytes. None is tensor-local or canonically
usable. Functional error reduction is zero, runtime correction cost is absent,
and artifact size remains unchanged only because no correction is inserted.
Larger BaseSteps create paid overhead. Separately, paid primary/correction
allocation shows that 3+5 can beat 4+4 at eight bits.

## DECISION

**E — no meaningful opportunity.**

## NEXT RECOMMENDED STEP

Do not promote slack-aware vBuf-ML representation work. Retain the independent
3+5 paid-budget result for future hidden-state testing.
