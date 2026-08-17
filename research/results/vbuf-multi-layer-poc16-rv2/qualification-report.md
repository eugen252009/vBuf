# POC16 RV2 Status

RV2 is reachable as `riscv64` and the exact persistent ranges required by the
x86 `blk.1..blk.3` run were replayed through the existing transport binary. The
pinned ggml RVV FP16 compute blocker remains unchanged.

- Exact ranges replayed: `102`.
- Attention and always-required FFN ranges: `33`.
- Selected routed expert slice ranges: `69`.
- Total transferred bytes: `135,599,104`.
- Single-source bytes: `135,599,104`.
- Striped bytes: `0`.
- Payload identity and teardown: PASS for all ranges.
- Unselected expert ranges and whole packed expert tensors: `0`.

```text
RV2_MULTI_LAYER_TRANSPORT_RESULT: PASS
RV2_MULTI_LAYER_COMPUTE_RESULT: NOT_EXECUTED
```
