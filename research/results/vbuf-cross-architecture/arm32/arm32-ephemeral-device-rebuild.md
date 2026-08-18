# ARM32 Ephemeral Device Rebuild

## Rebuild Inputs

The device is a throw-away Alpine ARM32 environment. The durable host inputs
are:

| Input | Durable location |
|---|---|
| exact source tree | workstation git commit `07becc3` |
| semantic bootstrap | `/home/eugen/arm32-qualification-backup/qwen32.semantic.vbuf` |
| bootstrap evidence | `arm32-remote-bootstrap-qualification.json` |
| qualification source | `scaffolding/arm32_bootstrap_qualification.rs` |
| HTTP Range server | `scaffolding/vbuf-arm32-range-server.py` |
| authoritative model | `research-models/Qwen3-32B-Q8_0.vbuf` |
| qualification binary | `/home/eugen/arm32-qualification-backup/arm32_bootstrap_qualification` |

The full model remains on the workstation only. It was never copied to ARM32.

## Exact Deployment

From the clean workstation checkout:

```sh
git archive --format=tar 07becc3 | ssh alpine \
  'mkdir -p /tmp/vbuf-arm32 && tar -xf - -C /tmp/vbuf-arm32'
scp /home/eugen/arm32-qualification-backup/qwen32.semantic.vbuf \
  alpine:/tmp/vbuf-arm32/qwen32.semantic.vbuf
```

This archive deployment has no `.git` directory on the target. The source
identity is established by the host-side `git archive` command and recorded as
`07becc3`.

## Build

The target source tree builds with the installed Alpine toolchain:

```sh
ssh alpine 'cd /tmp/vbuf-arm32/rust && \
  cargo build -p vbuf-ml --release \
    --bin vbuf-ml-semantic-parity \
    --bin vbuf-ml-semantic-bootstrap'
```

The ARM32-only qualification binary was temporary scaffolding, not a
production source change. Its source is preserved under `scaffolding/`.

## Reboot Checklist

After a reboot, the following can be reconstructed from the workstation:

- Board identity: Cubietruck Plus, Allwinner A83T, ARMv7, 32-bit userspace, 8 cores, 2,059,776 kB RAM.
- OS/toolchain: reinstall the packages and synchronize time with the recorded recipe.
- Source: recreate `/tmp/vbuf-arm32` from exact commit `07becc3` using `git archive`.
- Bootstrap: redeploy the host backup and verify its recorded SHA-256.
- HTTP source: start the preserved Range server against the host model artifact.
- Qualification: redeploy the preserved scaffolding, compile it with Cargo, and run the preserved commands.
- >4 GiB range: use SourceId 1, offset `6056603320`, length `5570560`.
- FFI materialization: the preserved scaffolding binds the downloaded range through the existing FFI boundary.

No unique qualification result depends on target-local package state, build
output, or temporary files after this backup.

## Known Limitations

The committed `consumer_ffi` test harness is `NOT_QUALIFIED` on this target
because Alpine ARM32 exposes `c_char = u8` while the test buffers use
`*mut i8`. The production FFI materialized-span boundary itself passed.

A later broader test batch was `BLOCKED` by temporary filesystem exhaustion
during linker output. This is an environment limit, not a behavioral failure.

No correction is made here; a future generic test-harness fix may use
`core::ffi::c_char`.
