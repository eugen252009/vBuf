# POC13 RV2 Full-Layer Transport Replay

RV2 full-layer compute remains `NOT_EXECUTED` because of the existing pinned
ggml RVV FP16 blocker. The complete required persistent range set was replayed
from the x86-qualified layer closure:

- 5 always-required tensor ranges: norm, router, and 3 shared tensors.
- 36 selected routed expert slices for A and B.
- Total ranges: `41`.
- Total logical and transferred bytes: `37658624`; all individual returned
  lengths and clean teardowns are recorded in this directory.
- Unselected expert ranges: `0`.
- Whole packed expert tensors: `0`.
- Source: single-source HTTP with RV2 local binding `192.168.188.50`.
- Striping: not used; no tensor required it for this replay.
