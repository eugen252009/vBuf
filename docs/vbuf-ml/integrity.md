# vBuf-ML optional integrity metadata (Step 14)

Integrity is optional downstream semantic information. It does not change
canonical v0.6 structural validity.

```text
canonical validation
→ canonical CheckedRange
→ optional expected digest
→ explicit runtime verification
→ consumer use
```

A structurally valid model may have no integrity metadata, may not yet have
been verified, or may later fail verification.

## Profile 0.1 contract

Bootstrap role `IntegrityMetadata` (role ID `4`) is optional. Its absence does
not invalidate a model. The integrity region is a non-continuing opaque byte
array with payload magic `VBINTG\0\0`, version `1`, and SHA-256 records.

Header: 20 bytes:

```text
magic[8], version:u16, flags:u16, entry_count:u32, reserved:u32
```

Entry: 40 bytes:

```text
canonical Key-ID:u16
canonical occurrence:u16
algorithm:u8 (1 = SHA-256)
flags:u8 (zero)
reserved:u16
expected digest[32]
```

Entries are sorted by `(Key-ID, occurrence)` and duplicate targets fail. The
maximum region is 16 MiB with at most 1,000,000 entries.

## Coverage and policy

The digest covers only the referenced canonical payload bytes. Canonical block
headers and padding are excluded, so legal physical relocation and alignment
changes preserve a digest when payload bytes remain unchanged.

`IntegrityMetadata::discover` resolves targets through canonical Key-ID and
occurrence and stores checked payload ranges. `verify_target` hashes only the
selected range. It never performs a global scan.

The file describes expected bytes; runtime policy decides whether to verify:

```text
never
at model open
at first target access
in preparation/background
only for selected semantic classes
```

No verification state is serialized. No digest authenticates a publisher or
protects against malicious replacement without an external trust mechanism.

## Cost and limitations

SHA-256 was selected as one stable, portable, established digest implementation
for profile 0.1. It detects accidental corruption and supplies a cryptographic
digest, but is not a signature or authentication system. Digest construction
necessarily reads covered bytes; verification is pay-as-requested.

No whole-file digest, per-padding digest, continuation-chain digest, checksum
registry, Nano dependency, finalization envelope, or backend policy is defined.
