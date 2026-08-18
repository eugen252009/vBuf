# Cross-Architecture Remote Bootstrap Qualification Matrix

This is the next research plan, not an executed deployment result. Only the
x86_64 column is qualified by the current milestone.

| Capability | x86_64 current | ARM32 | riscv64 | ARM64 |
|---|---|---|---|---|
| Generic vBuf parse | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Semantic bootstrap parse | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Discovery parity | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| u64 TensorRef | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Real >4 GiB logical offset | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| File RangeSource | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| HTTP RangeSource | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| External materialization | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Source-independent FFI handoff | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| FFI materialized-span binding | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Committed `consumer_ffi` test suite | PASS | PASS | NOT_QUALIFIED | NOT_QUALIFIED |
| Real ggml descriptor | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Bounded compute | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Full external token path | NOT_RUN | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |

`NOT_QUALIFIED` is deliberate. Parser portability, source portability, and
backend/compute portability are separate gates. A device may pass bootstrap,
range, and materialization while its pinned ggml compute remains unavailable
because of backend or compiler support.
