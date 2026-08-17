# POC11 RV2 Router-Driven Transport Replay

RV2 router computation was not executed because the pinned ggml RVV FP16
blocker remains unchanged. The x86-computed selected top-1 IDs were replayed
against the exact DeepSeek artifact ranges on RV2:

- Activation A selected ID `37`: gate `124602848/563200`, up
  `160647664/563200`, down `235868672/1622016`.
- Activation B selected ID `3`: gate `105454048/563200`, up
  `141498864/563200`, down `180720128/1622016`.

All six selected ranges returned exact requested bytes, used the configured
RV2 source binding, and reported `resources_after_teardown=0`. No unselected
expert range was requested. This is transport replay, not locally computed
routing.
