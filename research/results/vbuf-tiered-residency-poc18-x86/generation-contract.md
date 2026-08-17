# Generation Contract

- First backing admission: `NEW`.
- Second observed request: `YOUNG`.
- Third and later observed requests: `OLD`.
- Generation is reuse history, not hardware placement.
- Tier is independently `BACKING`, `WARM`, or `HOT`.
- Identity does not change during movement.
- No generation decay was required for this bounded four-epoch replay; teardown
  removes all entries and the contract tests movement/cleanup explicitly.
