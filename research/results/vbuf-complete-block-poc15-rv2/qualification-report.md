# POC15 RV2 Complete-Block Transport Replay

RV2 compute remains `NOT_EXECUTED` because the pinned ggml RVV FP16 blocker is
unchanged. Exact persistent ranges from the x86 complete-block qualification
were replayed using single-source transport:

- Six attention tensors: `2,960,384` bytes.
- Five always-required FFN tensors: `4,677,632` bytes.
- Eight unique selected-expert slices, three ranges each: `21,987,328` bytes.
- Total unique complete-block transport bytes: `29,625,344`.
- Ranges replayed: `35`.
- Single-source bytes: `29,625,344`.
- Striped bytes: `0`.
- Unselected expert ranges: `0`.
- Unrelated model/block ranges: `0`.
- Payload identity: PASS for all 35 ranges.
- Resources after teardown: `0` for all 35 ranges.
