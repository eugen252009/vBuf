# Project Agenda: vBuf (Vector-Buffer) ⚡

This document outlines the active roadmap, implementation status, and immediate task checklist for **vBuf** (Codename: **Kraftpaket**).

---

## 📌 Status Overview

vBuf is a high-performance, zero-copy binary serialization format designed for direct CPU vectorization (SIMD) and `mmap` reading.

*   **Current Specification:** [Specification v0.5-alpha](file:///home/eugen/projekte/vBuf/spec/spec_0.5-alpha.md)
*   **Target Alignment:** Dynamic Diamond Alignment ($16 \ll AShift$)
*   **Key Implementations:**
    *   TypeScript (Bun prototype) in [`ts/`](file:///home/eugen/projekte/vBuf/ts) *(Complete)*
    *   Rust in [`rust/`](file:///home/eugen/projekte/vBuf/rust) *(Complete)*
    *   C in [`c/`](file:///home/eugen/projekte/vBuf/c) *(Complete)*

---

## 🗺️ Roadmap & Milestones

### 🏁 Phase 1: Prototype Validation & Spec Alignment (Complete)
*   [x] Refactor TypeScript implementation to match `v0.5-alpha` specification.
*   [x] Validate Dynamic Diamond Alignment ($16 \ll AShift$) for payloads.
*   [x] Verify Atomic Block Header parsing (64-bit anchor) & Bit-width logic.
*   [x] Implement Overflow Mode (Bit 9) for quantities > 65K.
*   [x] Implement block chaining (Bit 8) logic with fallback recovery.

### 🦀 Phase 2: High-Performance Rust Library (Complete)
*   [x] Align the Rust workspace structure.
*   [x] Implement zero-copy decoding using the `zerocopy` crate.
*   [x] Set up criterion-based benchmarks comparing vBuf to FlexBuffers and FlatBuffers.
*   [x] Implement SSE/AVX2 vectorized scanning/filtering over payload streams.

### 🔌 Phase 3: Embedded C Implementation & Bindings (Complete)
*   [x] Complete C writer and reader conforming to `v0.5-alpha`.
*   [x] Create shared/static libraries and compile targets.
*   [x] Benchmark performance on lower-power embedded platforms.

---

## 📋 Immediate Action Items

### 1. Specification & Schema Validation
*   [x] Add spec test vectors (hex dumps of serialized cells) to [spec/](file:///home/eugen/projekte/vBuf/spec) for reference.

### 2. TypeScript (`ts/`)
*   [x] Update parser/writer with support for float mapping (`SEM`) and alignment constraints.
*   [x] Verify correct output via `debugger.ts` and `impl.ts` roundtrips.

### 3. Rust (`rust/`)
*   [x] Port the core writer and reader to Rust.
*   [x] Integrate unit tests matching the spec test vectors.

### 4. C (`c/`)
*   [x] Verify the `makefile` compile process.
*   [x] Ensure tests verify compliance with the same binary layout.
