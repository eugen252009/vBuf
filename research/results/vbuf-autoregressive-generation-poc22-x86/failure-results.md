# POC22 Failure Qualification

POC22's four-position success trace does not include recurrence-specific
failure injection. The existing POC19/POC21 failure and teardown contracts
remain applicable, but later-position embedding, block, and LM-head failure
cases were not executed in this qualification.

```text
LATER_POSITION_EMBEDDING_FAILURE_FAILS_CLOSED: NOT_EXECUTED
LATER_POSITION_BLOCK_FAILURE_FAILS_CLOSED: NOT_EXECUTED
LATER_POSITION_LM_HEAD_FAILURE_FAILS_CLOSED: NOT_EXECUTED
FAILED_REQUEST_WORKER_CLEANUP: NOT_EXECUTED
```
