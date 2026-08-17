# POC20 Audit

- Eviction currently occurs in `TensorResidencyStore::evict_one()` during insertion pressure.
- The existing policy is selectable LRU: ascending last request/use order with tensor-ref tie-break.
- Candidates are resident entries with `active_leases == 0`; active leases are hard exclusions.
- Runtime-visible generic facts are identity, resident size, lease count, request ordinal, observed request count, and reacquisition byte cost.
- `lookup()` updates recency; `note_request()` records request history before lookup.
- Materialization/source timing exists in the backing trace, but POC19 HTTP source cost is effectively uniform and no stable latency model is exposed to live replacement.
- Prefetch planning does not admit payloads and does not expose an oracle future request stream to replacement.
- Known runtime future: none used by COST_AWARE in this qualification.
- Oracle future: used only by offline MIN replay.
- The existing store is a single flat residency tier; no RAM/VRAM or HOT/WARM behavior is enabled.

## Gap

The old seam only compared last-use timestamps. It had no selectable generic policy object, no request-history counters, no source-cost field, and no policy overhead diagnostics. POC20 adds only those generic facts and keeps the existing linear candidate scan.
