# Data Movement Summary

For the required `8 MiB HOT + 32 MiB WARM` replay:

- Backing source bytes: `283,414,528`.
- Warm admissions: `348`.
- Warm hits: `252`.
- Hot hits: `84`.
- Promotions: `84`, `133,365,760` bytes.
- Demotions: `79`, `125,751,296` bytes.
- Warm drops: `302`, `243,493,888` bytes.
- Total inter-tier movement: `259,117,056` bytes.
- Total data movement: `542,531,584` bytes.
- Duplicate resident payload bytes: `0`.
- Dirty writeback bytes: `0`.

The tiered configuration does not reduce backing bytes relative to either the
8 MiB baseline or the equal-total-capacity 40 MiB flat control. It adds
substantial host-emulated inter-tier movement, so this first generational
policy is not promoted to runtime execution policy.
