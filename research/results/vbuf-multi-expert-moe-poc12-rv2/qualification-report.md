# POC12 RV2 Multi-Expert Transport Replay

RV2 compute was not executed because the pinned ggml RVV FP16 blocker remains
unchanged. The x86 POC12 router results were replayed as exact selected ranges.

- Activation A: experts `37,21,31,54,6,33`, 18 slices, `16490496` bytes.
- Activation B: experts `3,53,47,5,20,35`, 18 slices, `16490496` bytes.
- All 36 selected slice transports returned exact requested bytes.
- All transfers used single-source HTTP from `192.168.188.2` with RV2 local
  binding `192.168.188.50`.
- Unselected ranges requested: `0`.
- Whole packed expert tensors loaded: `0`.
- Resources after teardown: `0` for every range.
- Striped reads: `0`; single-source reads: `32980992` total replay bytes.
