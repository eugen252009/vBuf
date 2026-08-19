# POC22 RISC-V Qualification

POC22 ran natively on `riscv64` against
`DeepSeek-V2-Lite.IQ1_S.vbuf` after replacing the qualification harness's
eager whole-file heap copy with a read-only mmap.

Configuration:

```text
mode: full-stack
layers: 27
positions: 1
residency budget: 64 MiB
policy: LRU
```

Result:

```text
functional recurrence: PASS
reference token: 95626
runtime token: 95626
logits max absolute error: 0
peak resident tensor bytes: 66,603,008
peak active persistent bytes: 12,607,488
residency hits/misses/evictions: 1532 / 1503 / 686
tracked source bytes: 608,192,512
whole-model loading: NO
```

The run completed in approximately `654.0 s`. The board remained below the
device's 7.7 GiB RAM limit; the runtime's tensor working set stayed near the
configured 64 MiB budget. This is a native-runtime qualification, not yet an
OpenAI HTTP server integration or a multi-position performance qualification.
