# POC14 RV2 Attention Transport Replay

RV2 compute remains `NOT_EXECUTED` because of the pinned ggml RVV FP16
blocker. The six exact persistent attention ranges were replayed:

- norm: `8192` bytes
- q: `1228800` bytes
- kv-a: `230400` bytes
- kv-a norm: `2048` bytes
- kv-b: `409600` bytes
- output: `1081344` bytes

Total selected attention bytes and transferred bytes: `2960384`.
All six payloads returned exact bytes and clean teardown. No FFN/MoE range was
requested, and no whole model load occurred.
