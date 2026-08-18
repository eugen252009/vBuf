# ARM32 Reboot Survivability

```text
REBOOT_SURVIVABILITY: PASS
EPHEMERAL_TARGET_ONLY_IMPORTANT_FILES: NONE
```

If `cubietruck-plus` reboots and loses `/tmp`, `/root`, package state, and
build output, the qualification remains reconstructible from the workstation:

- Board identity and hardware are recorded in `device-baseline.txt`.
- The minimal Alpine package recipe and toolchain versions are recorded in `installed-for-vbuf-qualification.txt`.
- Commit `07becc3` and the exact `git archive` deployment are recorded in `arm32-ephemeral-device-rebuild.md` and `commands.log`.
- The semantic bootstrap is durably backed up at `/home/eugen/arm32-qualification-backup/qwen32.semantic.vbuf`.
- The full authoritative source remains at `research-models/Qwen3-32B-Q8_0.vbuf`.
- The HTTP Range server implementation and invocation are under `scaffolding/` and `commands.log`.
- The temporary ARM32 semantic/FFI qualification source is under `scaffolding/`.
- The small release binary is backed up at `/home/eugen/arm32-qualification-backup/arm32_bootstrap_qualification`.
- The exact offset, length, range hash, materialization result, and test counts are in `arm32-remote-bootstrap-qualification.json`.

The target still contains disposable copies of the source tree, bootstrap,
selected range, and release output, but none is unique after these backups.
The full 32B artifact is absent from the target.
