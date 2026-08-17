# Current Residency Policy Audit

- Ordering: `last_use` ascending, with tensor-ref ascending tie-break.
- Accesses: `lookup` updates `last_use`; admission assigns a new newest timestamp.
- Active leases: entries with nonzero leases are not eviction victims.
- Released entries: immediately eligible for eviction.
- Prefetch: planner output does not itself admit payloads; execution requests do.
- Admission: unconditional for every materialized tensor that fits.
- Diagnosis: the trace has 102 identities and all reload pairs exceed the 8 MiB
  unique-intervening-byte capacity; the dominant limit is sequential capacity,
  not an incorrect LRU ordering or one-shot admission pollution.
- Oracle: MIN reduces reload bytes by only 11.0554%, so no speculative runtime
  policy is implemented in POC17.
