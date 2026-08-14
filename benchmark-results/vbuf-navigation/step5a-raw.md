# Step 5A raw navigation qualification

This report is generated from the CSV; timings are measurements, not wire decisions.

- CPU: `unreported`
- Python validator: `3.13.5`
- Raw rows: 12480; samples/group: 20; BaseSteps: 8, 16, 32, 64, 128, 256 bytes

## Median nanoseconds by operation (all layouts pooled per BaseStep)

| BaseStep | Operation | Variant | Median ns |
|---:|---|---|---:|
| 8 | canonical-block-traversal | canonical | 8,665.0 |
| 8 | canonical-validation | parse-and-describe | 2,775.0 |
| 8 | key-lookup | canonical-linear-scan | 535.0 |
| 8 | key-lookup | directory-binary-search | 430.0 |
| 8 | nano-construction | reconstruct-from-validated-canonical | 1,450.0 |
| 8 | nano-deployment | embedded-load | 20.0 |
| 8 | nano-deployment | reconstructed-cache-load | 20.0 |
| 8 | nth-start | nano+checkpoint-4096 | 245.0 |
| 8 | nth-start | nano+checkpoint-512 | 295.0 |
| 8 | nth-start | nano+checkpoint-65536 | 245.0 |
| 8 | parallel-payload-sum | canonical-rayon | 24,990.0 |
| 8 | parallel-start-enumeration | nano-rayon | 14,830.0 |
| 8 | physical-start-enumeration | nano | 115.0 |
| 16 | canonical-block-traversal | canonical | 9,390.0 |
| 16 | canonical-validation | parse-and-describe | 3,935.0 |
| 16 | key-lookup | canonical-linear-scan | 520.0 |
| 16 | key-lookup | directory-binary-search | 415.0 |
| 16 | nano-construction | reconstruct-from-validated-canonical | 950.0 |
| 16 | nano-deployment | embedded-load | 20.0 |
| 16 | nano-deployment | reconstructed-cache-load | 20.0 |
| 16 | nth-start | nano+checkpoint-4096 | 245.0 |
| 16 | nth-start | nano+checkpoint-512 | 295.0 |
| 16 | nth-start | nano+checkpoint-65536 | 245.0 |
| 16 | parallel-payload-sum | canonical-rayon | 24,745.0 |
| 16 | parallel-start-enumeration | nano-rayon | 14,880.0 |
| 16 | physical-start-enumeration | nano | 125.0 |
| 32 | canonical-block-traversal | canonical | 9,255.0 |
| 32 | canonical-validation | parse-and-describe | 6,245.0 |
| 32 | key-lookup | canonical-linear-scan | 520.0 |
| 32 | key-lookup | directory-binary-search | 430.0 |
| 32 | nano-construction | reconstruct-from-validated-canonical | 685.0 |
| 32 | nano-deployment | embedded-load | 20.0 |
| 32 | nano-deployment | reconstructed-cache-load | 20.0 |
| 32 | nth-start | nano+checkpoint-4096 | 245.0 |
| 32 | nth-start | nano+checkpoint-512 | 310.0 |
| 32 | nth-start | nano+checkpoint-65536 | 245.0 |
| 32 | parallel-payload-sum | canonical-rayon | 25,615.0 |
| 32 | parallel-start-enumeration | nano-rayon | 14,450.0 |
| 32 | physical-start-enumeration | nano | 115.0 |
| 64 | canonical-block-traversal | canonical | 10,280.0 |
| 64 | canonical-validation | parse-and-describe | 9,920.0 |
| 64 | key-lookup | canonical-linear-scan | 515.0 |
| 64 | key-lookup | directory-binary-search | 430.0 |
| 64 | nano-construction | reconstruct-from-validated-canonical | 470.0 |
| 64 | nano-deployment | embedded-load | 20.0 |
| 64 | nano-deployment | reconstructed-cache-load | 20.0 |
| 64 | nth-start | nano+checkpoint-4096 | 245.0 |
| 64 | nth-start | nano+checkpoint-512 | 355.0 |
| 64 | nth-start | nano+checkpoint-65536 | 245.0 |
| 64 | parallel-payload-sum | canonical-rayon | 23,055.0 |
| 64 | parallel-start-enumeration | nano-rayon | 14,105.0 |
| 64 | physical-start-enumeration | nano | 110.0 |
| 128 | canonical-block-traversal | canonical | 15,665.0 |
| 128 | canonical-validation | parse-and-describe | 19,725.0 |
| 128 | key-lookup | canonical-linear-scan | 510.0 |
| 128 | key-lookup | directory-binary-search | 415.0 |
| 128 | nano-construction | reconstruct-from-validated-canonical | 370.0 |
| 128 | nano-deployment | embedded-load | 20.0 |
| 128 | nano-deployment | reconstructed-cache-load | 20.0 |
| 128 | nth-start | nano+checkpoint-4096 | 245.0 |
| 128 | nth-start | nano+checkpoint-512 | 300.0 |
| 128 | nth-start | nano+checkpoint-65536 | 245.0 |
| 128 | parallel-payload-sum | canonical-rayon | 24,780.0 |
| 128 | parallel-start-enumeration | nano-rayon | 14,260.0 |
| 128 | physical-start-enumeration | nano | 115.0 |
| 256 | canonical-block-traversal | canonical | 19,170.5 |
| 256 | canonical-validation | parse-and-describe | 37,220.0 |
| 256 | key-lookup | canonical-linear-scan | 520.0 |
| 256 | key-lookup | directory-binary-search | 415.0 |
| 256 | nano-construction | reconstruct-from-validated-canonical | 345.0 |
| 256 | nano-deployment | embedded-load | 20.0 |
| 256 | nano-deployment | reconstructed-cache-load | 20.0 |
| 256 | nth-start | nano+checkpoint-4096 | 245.0 |
| 256 | nth-start | nano+checkpoint-512 | 300.0 |
| 256 | nth-start | nano+checkpoint-65536 | 245.0 |
| 256 | parallel-payload-sum | canonical-rayon | 24,960.0 |
| 256 | parallel-start-enumeration | nano-rayon | 13,550.0 |
| 256 | physical-start-enumeration | nano | 125.0 |

## Measured facts and conservative decisions

- **Measured fact:** all six frozen BaseSteps and all eight generic layouts completed with stable metadata and 20 samples per operation.
- **Measured fact:** Nano bytes and padding vary with BaseStep and topology; see the raw `file_bytes`, `payload_bytes`, `nano_bytes`, and checkpoint/directory columns.
- **Inference:** these in-memory timings do not establish a universal BaseStep winner, nor do they include OS-controlled cold-cache state.
- **Decision:** BaseStep remains a legal generic tuning input; no BaseStep is promoted or removed by this run.
- **Decision:** Nano, checkpoints, and region-directory paths remain `POSSIBLE BUT NOT YET JUSTIFIED` pending independent end-to-end qualification including construction, persistence, validation, and equivalent lookup work.
- **Decision:** no experimental artifact becomes normative and no finalization envelope is added in Step 5A.

## Environment limitations

The Rust runner records compiler/build context in the repository; this validation report intentionally does not infer cache, fault, SIMD, or pointer-alignment claims that were not instrumented.
