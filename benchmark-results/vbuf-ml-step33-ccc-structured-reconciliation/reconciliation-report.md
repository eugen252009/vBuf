# vBuf_4 structured CCC evidence reconciliation

## Git provenance

- Pre-merge current HEAD: `1ea629b8fabf882065472604065e68dee5e51ea0`
- vBuf_4 original HEAD / merge base: `33a4d037015b09394b762d5df159dad6d42f8643`
- vBuf_4 preservation commit: `28588a048a02ee3f4df33530dff8ee789e290114`
- No-ff merge: `53afb10b26af6a57a8a4ff6bf7c80728c850f39e`
- Histories were preserved as separate parents; no squash or rebase.

## Imported evidence

The imported branch evaluated 111 structured-baseline/residual-alphabet candidates on `blk.0.attn_k.weight` from the same verified Q8_0 model. It used the correct `(1024,5120)` NumPy orientation. Imported artifacts remain unchanged.

## Agreements

- vBuf_4 and Stage-1 independently find zero/tensor anchors tied and row/column anchors non-beneficial on blk.0.attn_k.weight.
- vBuf_4 and Stage-2 both place plain learned/geometric C3 near approximately 0.20 action/relative error.
- No evidence supports contextual prediction as the source of low-bit accuracy.
- No production representation is justified.

## Scope distinctions

- vBuf_4 evaluates blk.0, while Stage-2 real hidden states evaluate blk.32.
- vBuf_4 uses Gaussian probes only; it does not supersede real-hidden-state evidence.
- vBuf_4 G2/G3 include an exact zero state; its G4 no-zero control is the closer analogue to Stage-2/parallel mid-riser C3.
- vBuf_4 predates IQ controls; Stage-2 IQ2_XS remains decisive for numerical domination.

## Structured predictor result

On the untouched vBuf_4 TEST split, B1–B5 change residual RMS by at most about 0.12%; row, column, and row+column predictors do not reduce residual energy. Stage-1 independently shows the same pattern on this exact tensor: zero/tensor anchors tie, while row/column metadata slightly worsens C3 error.

## Alphabet result

vBuf_4 zero-state power families G2/G3 trail free Lloyd-Max by approximately 5.7–13%. Its no-zero G4 is better than those controls and is the proper analogue of the mid-riser C3 tested elsewhere. This supports a nonuniform scalar alphabet, not contextual correction. Stage-2 IQ controls still dominate C3 under rate × quality.

## Canonical and functional limits

vBuf_4 uses pinned canonical controls and three Gaussian W*x probes. It has no real hidden-state evidence and therefore does not supersede Stage-2. Its best CCC-family W*x point is about 0.2125 versus Q3_K about 0.1545 on its probes.

## Combined classifications

Structured baseline: `STRUCTURED_BASELINE_REJECTED`
Geometry: `GEOMETRY_APPROXIMATES_LEARNED`
C3 numerical rate/quality: `C3_NUMERICALLY_DOMINATED`
C3 direct apply: `C3_DIRECT_APPLY_FEASIBILITY_CONFIRMED`
Combined direction: `STOP_C3`

C4+0.1% tail remains unchanged and visible: 4.0481 bpw, 0.0556 real W*x, direct apply unproven.

## Recommendation

`STOP_C3`

The independent vBuf_4 evidence strengthens rejection of the contextual/structured-baseline premise on attention key projections. Preserve it as bounded single-tensor evidence; do not broaden, implement, or modify production architecture.

## Stop

No vBuf, vBuf-ML, quantizer, native-kernel, or production-runtime changes were made.
