# RV2 MoE Transport-Only POC10

The pinned RVV ggml compute blocker remains unchanged, so this qualification
uses the existing POC9 transport executable against the exact selected Expert 0
slice ranges.

- Gate: `103764448`, `563200` bytes, hash `b475e17840345e07`
- Up: `139809264`, `563200` bytes, hash `3dc1f18d4f338087`
- Down: `175854080`, `1622016` bytes, hash `fcb2d9b265c48e55`
- All selected slices returned exact bytes and reported clean teardown.
- Local IP bindings were verified on the configured RV2 source paths.
- Unselected expert ranges were not requested.
