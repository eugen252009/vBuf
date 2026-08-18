# POC22 Recurrence Audit

POC21 passed two independently supplied token IDs at positions 0 and 1. The
driver did not feed its output back into embedding and allocated each runtime
state slot with capacity 2.

POC22 reuses the qualified `run_embedding`, `run_sequence`, and output-head
path. It supplies one current token ID per iteration, passes the loop position
to every block, appends one K/V entry per layer, compares the independent
reference graph, selects greedy argmax, and assigns the runtime result as the
next iteration's input. Actual and reference state remain separate. The same
residency/materializer instance is reused across positions.

The current qualification uses seed token `0`, four generated positions, and
the existing `COST_AWARE` capacity of `268435456` bytes. It is x86-only.

The reference in this native driver is the existing independent vBuf reference
graph used by POC21. A pinned llama.cpp/GGUF autoregressive oracle was not
available in the current build environment and is therefore recorded as a
remaining external-oracle gap rather than claimed as executed here.
