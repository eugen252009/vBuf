# Step 31X Cancellation-Aware Serial Admission

Status: **X86_64 IMPLEMENTED AND PHYSICALLY QUALIFIED; SERIAL INFERENCE
PRESERVED**.

Target: Linux x86_64 workstation. Android, ADB, Orange Pi, physical RISC-V,
and pinned-GGML equivalence were not required or used.

The qualification used the available ggml checkout at commit
`a97123e497968f3440264c0464a7adc7c999c027`. The canonical pinned ggml revision
remains unavailable. The change does not modify the vBuf wire format, vBuf-ML
materialization, residency policy, KV semantics, or GGML execution.

## Admission Contract

Accepted generation requests now enter an explicit lifecycle:

```text
WAITING -> ADMITTED -> ACTIVE -> COMPLETED | FAILED | CANCELLED
```

The fair admission gate still uses accepted-connection order and grants at most
one generation lease. A waiting worker polls its client socket every 50 ms and
checks for peer close, while also observing server shutdown. A disconnected or
shutdown waiter is removed from the waiting-generation set before admission,
and the next eligible generation proceeds without a FIFO hole.

Control endpoints remain outside the inference gate. HTTP parsing and bounded
request validation occur before generation admission, but SSE headers and
runtime entry occur only after the generation lease is acquired.

The handler defers gate release until its terminal diagnostics are emitted.
This prevents the next generation from entering runtime while the current
request is still reporting `active_cancellations_after`, stream, request, and
lease counters. Pre-admission cancellation notification is likewise deferred
until the cancelled request's terminal record is emitted.

## Qualification Matrix

The Step 31X harness passed all scenarios:

| Scenario | Result |
|---|---|
| queued non-stream disconnect | `PASS`, runtime entry count `0` |
| queued stream disconnect | `PASS`, runtime entry count `0` |
| middle waiter cancellation | `PASS`, FIFO survivors |
| first waiter cancellation | `PASS`, FIFO survivors |
| last waiter cancellation | `PASS`, FIFO survivors |
| all queued waiters cancellation | `PASS` |
| faulted generation with cancelled follower | `PASS` |
| control requests with active/waiting generations | `PASS` |
| shutdown with waiting generations | `PASS` |
| admission/cancellation race | `PASS`, one runtime winner per repetition |

The full evidence is
`/tmp/opencode/vbuf-step31x-cancellable-admission-fixed.json`.

The persistent lifecycle regression also passed with 20 sequential requests,
streaming, cancellation recovery, failure recovery, bounded residency, source
reuse, health during generation, serial concurrency, and clean shutdown. Its
evidence is `/tmp/opencode/vbuf-step31x-lifecycle-fixed.json`.

The Step 31V serial-queue regression and Step 31W control-plane regression both
passed after the change. Their evidence is:

```text
/tmp/opencode/vbuf-step31v-serial-queue-fixed.json
/tmp/opencode/vbuf-step31w-control-plane-fixed.json
```

## Boundary And Preserved Behavior

This step covers requests already accepted by the bounded connection-worker
pool. It does not claim to cancel a client that disconnects while its request
is still only in the kernel listen backlog; the Step 31V backlog gap remains a
separate limitation.

The following remain unchanged:

- exactly one active `VbufGenerationSession::run`;
- no application generation queue;
- no batching, slots, interleaving, or parallel inference;
- no backend or GGML loader ownership of runtime acquisition;
- bounded worker count and bounded connection admission;
- control-plane responsiveness during serial inference.

Evidence harness:
`integrations/ggml/tests/vbuf_step31x_cancellable_admission_test.py`.
