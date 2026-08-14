# Step 5A raw navigation qualification

This report is generated from the CSV; timings are measurements, not wire decisions.

- CPU: `unreported`
- Python validator: `3.13.5`
- Raw rows: 17280; groups: 864; samples/group: 20; operation variants: 18; BaseSteps: 8, 16, 32, 64, 128, 256 bytes

## Median nanoseconds by operation (all layouts pooled per BaseStep)

| BaseStep | Operation | Variant | Median ns |
|---:|---|---|---:|
| 8 | canonical-block-traversal | canonical | 9,260.0 |
| 8 | canonical-validation | parse-and-describe | 2,660.0 |
| 8 | directory-construction | sorted-validated-ranges | 3,280.0 |
| 8 | key-lookup | canonical-linear-scan | 25,950.0 |
| 8 | key-lookup | directory-binary-search | 1,360.0 |
| 8 | nano-construction | reconstruct-from-validated-canonical | 1,400.0 |
| 8 | nano-deployment | embedded-load | 470.0 |
| 8 | nano-deployment | local-reconstruction | 4,335.0 |
| 8 | nano-deployment | reconstructed-cache-load | 510.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 589,800.0 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-512 | 507,113.5 |
| 8 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 631,520.0 |
| 8 | nth-start-plus-next-boundary | raw-nano-linear-select | 3,377,210.5 |
| 8 | parallel-payload-sum | canonical-rayon | 24,935.0 |
| 8 | parallel-payload-sum | dynamic-queue-control | 24,245.5 |
| 8 | parallel-payload-sum | nano-guided-rayon | 269,312.0 |
| 8 | parallel-start-enumeration | nano-rayon | 26,425.0 |
| 8 | physical-start-enumeration | nano | 4,105.0 |
| 16 | canonical-block-traversal | canonical | 8,650.0 |
| 16 | canonical-validation | parse-and-describe | 3,750.0 |
| 16 | directory-construction | sorted-validated-ranges | 3,280.0 |
| 16 | key-lookup | canonical-linear-scan | 25,945.0 |
| 16 | key-lookup | directory-binary-search | 1,350.0 |
| 16 | nano-construction | reconstruct-from-validated-canonical | 950.0 |
| 16 | nano-deployment | embedded-load | 400.0 |
| 16 | nano-deployment | local-reconstruction | 4,300.0 |
| 16 | nano-deployment | reconstructed-cache-load | 400.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 381,938.0 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-512 | 291,587.5 |
| 16 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 395,918.0 |
| 16 | nth-start-plus-next-boundary | raw-nano-linear-select | 1,904,010.0 |
| 16 | parallel-payload-sum | canonical-rayon | 25,745.0 |
| 16 | parallel-payload-sum | dynamic-queue-control | 25,145.5 |
| 16 | parallel-payload-sum | nano-guided-rayon | 264,287.0 |
| 16 | parallel-start-enumeration | nano-rayon | 26,111.0 |
| 16 | physical-start-enumeration | nano | 3,305.0 |
| 32 | canonical-block-traversal | canonical | 8,845.0 |
| 32 | canonical-validation | parse-and-describe | 5,680.0 |
| 32 | directory-construction | sorted-validated-ranges | 3,265.0 |
| 32 | key-lookup | canonical-linear-scan | 25,950.0 |
| 32 | key-lookup | directory-binary-search | 1,365.0 |
| 32 | nano-construction | reconstruct-from-validated-canonical | 640.0 |
| 32 | nano-deployment | embedded-load | 400.0 |
| 32 | nano-deployment | local-reconstruction | 4,305.5 |
| 32 | nano-deployment | reconstructed-cache-load | 400.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 232,067.0 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-512 | 191,551.5 |
| 32 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 317,897.0 |
| 32 | nth-start-plus-next-boundary | raw-nano-linear-select | 943,536.5 |
| 32 | parallel-payload-sum | canonical-rayon | 24,765.0 |
| 32 | parallel-payload-sum | dynamic-queue-control | 24,315.0 |
| 32 | parallel-payload-sum | nano-guided-rayon | 238,217.0 |
| 32 | parallel-start-enumeration | nano-rayon | 25,830.5 |
| 32 | physical-start-enumeration | nano | 3,280.0 |
| 64 | canonical-block-traversal | canonical | 9,450.5 |
| 64 | canonical-validation | parse-and-describe | 10,000.0 |
| 64 | directory-construction | sorted-validated-ranges | 3,260.0 |
| 64 | key-lookup | canonical-linear-scan | 25,940.0 |
| 64 | key-lookup | directory-binary-search | 1,360.0 |
| 64 | nano-construction | reconstruct-from-validated-canonical | 455.0 |
| 64 | nano-deployment | embedded-load | 400.0 |
| 64 | nano-deployment | local-reconstruction | 4,100.5 |
| 64 | nano-deployment | reconstructed-cache-load | 400.0 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 130,400.5 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-512 | 99,521.0 |
| 64 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 288,182.0 |
| 64 | nth-start-plus-next-boundary | raw-nano-linear-select | 531,584.0 |
| 64 | parallel-payload-sum | canonical-rayon | 26,220.0 |
| 64 | parallel-payload-sum | dynamic-queue-control | 25,160.0 |
| 64 | parallel-payload-sum | nano-guided-rayon | 235,107.0 |
| 64 | parallel-start-enumeration | nano-rayon | 28,140.0 |
| 64 | physical-start-enumeration | nano | 3,280.0 |
| 128 | canonical-block-traversal | canonical | 15,955.0 |
| 128 | canonical-validation | parse-and-describe | 19,370.5 |
| 128 | directory-construction | sorted-validated-ranges | 3,280.0 |
| 128 | key-lookup | canonical-linear-scan | 25,950.0 |
| 128 | key-lookup | directory-binary-search | 1,390.0 |
| 128 | nano-construction | reconstruct-from-validated-canonical | 415.0 |
| 128 | nano-deployment | embedded-load | 400.0 |
| 128 | nano-deployment | local-reconstruction | 2,380.0 |
| 128 | nano-deployment | reconstructed-cache-load | 400.0 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 99,600.5 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-512 | 64,745.5 |
| 128 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 272,342.0 |
| 128 | nth-start-plus-next-boundary | raw-nano-linear-select | 304,597.5 |
| 128 | parallel-payload-sum | canonical-rayon | 26,526.0 |
| 128 | parallel-payload-sum | dynamic-queue-control | 26,275.0 |
| 128 | parallel-payload-sum | nano-guided-rayon | 225,542.0 |
| 128 | parallel-start-enumeration | nano-rayon | 26,705.0 |
| 128 | physical-start-enumeration | nano | 3,280.0 |
| 256 | canonical-block-traversal | canonical | 15,325.0 |
| 256 | canonical-validation | parse-and-describe | 36,540.5 |
| 256 | directory-construction | sorted-validated-ranges | 3,275.0 |
| 256 | key-lookup | canonical-linear-scan | 25,935.0 |
| 256 | key-lookup | directory-binary-search | 1,750.0 |
| 256 | nano-construction | reconstruct-from-validated-canonical | 345.0 |
| 256 | nano-deployment | embedded-load | 400.0 |
| 256 | nano-deployment | local-reconstruction | 1,350.0 |
| 256 | nano-deployment | reconstructed-cache-load | 400.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-4096 | 85,805.5 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-512 | 51,691.0 |
| 256 | nth-start-plus-next-boundary | nano+checkpoint-65536 | 172,446.0 |
| 256 | nth-start-plus-next-boundary | raw-nano-linear-select | 193,981.5 |
| 256 | parallel-payload-sum | canonical-rayon | 25,920.0 |
| 256 | parallel-payload-sum | dynamic-queue-control | 25,170.0 |
| 256 | parallel-payload-sum | nano-guided-rayon | 227,627.0 |
| 256 | parallel-start-enumeration | nano-rayon | 26,665.5 |
| 256 | physical-start-enumeration | nano | 3,280.0 |

## Measured facts and conservative decisions

- **Measured fact:** all six frozen BaseSteps and all eight generic layouts completed with stable metadata and 20 samples per operation.
- **Measured fact:** exact header and padding bytes are recorded separately from payload, Nano, checkpoint, and directory bytes in the raw CSV.
- **Inference:** these in-memory timings do not establish a universal BaseStep winner, nor do they include OS-controlled cold-cache state.
- **Decision:** BaseStep remains a legal generic tuning input; no BaseStep is promoted or removed by this run.
- **Decision:** Nano, checkpoints, and region-directory paths remain `POSSIBLE BUT NOT YET JUSTIFIED` pending independent end-to-end qualification including construction, persistence, validation, and equivalent lookup work.
- **Decision:** no experimental artifact becomes normative and no finalization envelope is added in Step 5A.

## Environment limitations

The Rust runner records compiler/build context in the repository; this validation report intentionally does not infer cache, fault, SIMD, or pointer-alignment claims that were not instrumented.
