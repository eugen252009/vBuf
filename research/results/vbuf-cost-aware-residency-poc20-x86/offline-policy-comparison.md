# Offline POC20 Policy Comparison

The trace is the real POC19 8-block request trace.
COST_AWARE uses only observed request count, recency, size, and byte-based reacquisition cost.

| Capacity | Policy | Source bytes | Reload bytes | Reload events | Evictions | Decisions | Candidates |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 MiB | lru | 750665728 | 516276224 | 645 | 920 | 920 | 9221 |
| 8 MiB | cost-aware | 750223360 | 515833856 | 573 | 825 | 825 | 24801 |
| 8 MiB | min | 729744384 | 495354880 | 613 | 887 | 887 | 10274 |
| 64 MiB | lru | 750665728 | 516276224 | 645 | 849 | 849 | 69712 |
| 64 MiB | cost-aware | 730408960 | 496019456 | 528 | 704 | 704 | 72992 |
| 64 MiB | min | 553674752 | 319285248 | 388 | 598 | 598 | 50220 |
| 128 MiB | lru | 750665728 | 516276224 | 645 | 765 | 765 | 126462 |
| 128 MiB | cost-aware | 578158592 | 343769088 | 368 | 471 | 471 | 82961 |
| 128 MiB | min | 352397312 | 118007808 | 124 | 263 | 263 | 44431 |
| 256 MiB | lru | 234389504 | 0 | 0 | 0 | 0 | 0 |
| 256 MiB | cost-aware | 234389504 | 0 | 0 | 0 | 0 | 0 |
| 256 MiB | min | 234389504 | 0 | 0 | 0 | 0 | 0 |

## Acceptance

- 64 MiB: COST_AWARE reload bytes 496019456 versus LRU 516276224; headroom capture 10.28%.
- 128 MiB: COST_AWARE reload bytes 343769088 versus LRU 516276224; headroom capture 43.31%.
