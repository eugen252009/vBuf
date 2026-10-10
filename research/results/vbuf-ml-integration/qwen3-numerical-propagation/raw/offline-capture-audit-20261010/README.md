# Offline native-AV evidence audit artifacts

Generated from committed/recorded evidence only. The audit script does not run model inference, access the network, modify fixture statuses, or launch CUDA work. `CUDA_VISIBLE_DEVICES` was empty during the C++ reference replay and focused CTest.

Files:

- `fixture-inventory.csv`: 70 evidence/control records with scope, input/topology equivalence, status, repeat count, and source path.
- `capture-integrity.csv`: 18 boundary-capture metadata/binary checks with expected geometry and exact size.
- `audit-summary.json`: aggregate counts and stopped-matrix totals.
- `reference-replay.log`: full CPU reference evaluator output for capacity-512 prefill and decode captures.
- `capture-tamper-negative-control.txt`: expected evaluator failure after flipping one bit in a temporary copy of a captured softmax value; original evidence is untouched.
- `focused-ctest.log`: focused test result.
- `audit-command.txt`, `environment.txt`, and `sha256sums.txt`: command/environment and artifact integrity records.

The inventory's PASS rows include diagnostic controls, same-input checks, and repeated records; they are not independent full-model qualifications. The original matrix's 19 unrun rows remain `NOT_TESTED`. See [`../../../qwen3-native-av-offline-failure-audit.md`](../../../qwen3-native-av-offline-failure-audit.md) for interpretation and limitations.
