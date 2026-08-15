# vBuf_2 geometric CCC and C4 reconciliation

## Git provenance

- Pre-merge current HEAD: `df0086304fea1d1556138bf798fde36eb16adf50`
- vBuf_2 original HEAD / merge base: `33a4d037015b09394b762d5df159dad6d42f8643`
- vBuf_2 preservation commit: `da93e439c7836837f13c11341767142d892fad68`
- No-ff merge: `51311440ee955acdb452202f17ab5d01536767e9`
- Independent parentage is preserved; no squash or rebase.

## Imported evidence

The branch contains an early bounded geometric search and a later C4 hard gate on `blk.0.attn_k.weight`. The later gate adds pinned canonical Q/IQ controls and 69 real F32 vectors captured at `attn_norm-0`. Historical artifacts remain unchanged.

## Cross-layer real W*x comparison

Candidate | vBuf_2 layer 0 | Stage-2 layer 32
---|---:|---:
free_c3 | 0.256696 | 0.201591
geometric_c3 | 0.226940 | 0.214043
free_c4 | 0.153030 | 0.147348
geometric_c4 | 0.199735 | 0.163440
Q3_K | 0.041210 | 0.068240
IQ3_XXS | 0.061975 | 0.100572
Q4_K | 0.019615 | 0.033321
IQ4_XS | 0.021040 | 0.035654

## Agreements

- Plain free C4 has real W*x error near 0.15 in both layers (0.1530 layer 0; 0.1473 layer 32).
- Geometric C4 is worse than free C4 on validation, test weights, and real hidden states.
- Canonical Q/IQ formats strongly outperform plain free/geometric C4.
- C3 remains numerically dominated.

## Scope boundaries

- vBuf_2 layer-0 and Stage-2 layer-32 values are independent cross-layer evidence, not repeated samples.
- vBuf_2 did not test Stage-2 fixed 0.1% sparse FP16 C4 tail.
- The geometric-qualification precursor had canonical formats blocked; the later C4 hard gate supersedes that limitation without rewriting it.

## C4 interpretation

A strengthened free 16-level fitter beats geometric C4 on FIT, validation, TEST, and real hidden states. Canonical Q4_K/IQ4_XS are dramatically better functionally. The earlier slight geometric lead was a weak free-control fit and is superseded within the imported branch by its hard gate.

Stage-2 C4 plus a fixed 0.1% sparse FP16 residual tail remains a distinct candidate. vBuf_2 tests no sparse exception payload or index stream, so it neither confirms nor falsifies that point.

## Classifications

C3: `STOP_C3`
Free versus geometric C4: `C4_FREE_DOMINATES`
Plain C4 versus canonical: `C4_CANONICALLY_DOMINATED`
Sparse C4 tail: `C4_SPARSE_TAIL_SURVIVES_UNREPLICATED`
C4 direct apply: `C4_DIRECT_APPLY_UNPROVEN`
Overall: `KEEP_C4_SPARSE_TAIL_AS_EVIDENCE_ONLY`

## Recommendation

Do not promote C3, plain C4, or geometric C4. Preserve the Stage-2 sparse C4-tail point as the only numerical CCC survivor, explicitly unreplicated and direct-apply unproven. Do not implement it in this preservation/reconciliation step.

## Stop

No vBuf, vBuf-ML, native-kernel, quantizer, or production-runtime changes were made.
