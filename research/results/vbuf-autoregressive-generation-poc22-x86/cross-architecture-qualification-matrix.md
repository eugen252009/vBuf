# Cross-Architecture Remote Bootstrap Qualification Matrix

The x86_64, ARM32, and ARM64 storage/source columns reflect executed
qualification results. RISC-V remains unqualified; backend/compute rows remain
separate from parser, source, and materialization portability.

| Capability | x86_64 current | ARM32 | riscv64 | ARM64 |
|---|---|---|---|---|
| Generic vBuf parse | PASS | PASS | PASS | PASS |
| Semantic bootstrap parse | PASS | PASS | PASS | PASS |
| Discovery parity | PASS | PASS | PASS | PASS |
| u64 TensorRef | PASS | PASS | PASS | PASS |
| Real >4 GiB logical offset | PASS | PASS | PASS | PASS |
| File RangeSource | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| HTTP RangeSource | PASS | PASS | PASS | PASS |
| External materialization | PASS | PASS | PASS | PASS |
| Source-independent FFI handoff | PASS | PASS | PASS | PASS |
| FFI materialized-span binding | PASS | PASS | PASS | PASS |
| Committed `consumer_ffi` test suite | PASS | PASS | PASS | PASS |
| Real ggml descriptor | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Bounded compute | PASS | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |
| Full external token path | NOT_RUN | NOT_QUALIFIED | NOT_QUALIFIED | NOT_QUALIFIED |

`NOT_QUALIFIED` is deliberate. Parser portability, source portability, and
backend/compute portability are separate gates. A device may pass bootstrap,
range, and materialization while its pinned ggml compute remains unavailable
because of backend or compiler support.
