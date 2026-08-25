# Hugging Face Safetensors Import Checklist

## Plan

- [x] Resolve and persist repository, requested revision, and immutable commit.
- [x] Fetch config/index and bounded Safetensors headers only.
- [x] Validate index ownership against actual shard headers.
- [x] Use checked u64 source and destination arithmetic.
- [x] Complete canonical destination layout before bulk payload acquisition.
- [x] Report dtype and declarative semantic gaps without executing remote code.

## Execute

- [x] Pre-size one canonical `.vbuf.partial` output.
- [x] Use bounded coalesced source windows and positioned scatter writes.
- [x] Reject non-206, short, mismatched, or over-limit range responses.
- [x] Publish completion only after destination durability and journal update.
- [x] Resume only with identical source identity, plan digest, and destination.
- [x] Atomically publish the final canonical filename after validation.

## Qualification Boundary

- [x] Offline header, sharding, parity, retry, resume, space, and u64 tests.
- [x] Local HTTP redirect/range smoke through the production transport.
- [x] Real public single-file Safetensors smoke.
- [ ] Real public sharded Safetensors smoke.
- [ ] Tokenizer persistence parity for a self-contained runtime artifact.
- [ ] Architecture-specific expert-bank provenance for a supported MoE profile.
- [ ] Plan-only qualification against the authorized 100--200+ GiB repository.

Safetensors remains an import-time source. The normal vBuf-ML runtime consumes
only the finalized canonical vBuf artifact.
