---
name: llvm-langref
description: >
  Language Reference for LLVM IR. Use this to understand semantics and behavior
  of LLVM instructions and generic intrinsics. **Avoid** using this for
  target-specific intrinsics (e.g. AMDGCN, NVVM, ARM/AArch64, x86) as they are
  not described here. Trigger phrases include "check LLVM IR semantics", "verify
  what this means in LLVM IR". Also applicable for reviewing generated or
  produced LLVM IR.
---

In `reference/langref.md`, you will find the language reference for LLVM IR
(2026-06-08). The file is rather large (1.2M) so avoid reading all of it,
instead `grep` it with specific instructions and look at surrounding lines for
context information. Also read the higher-level sections of the file for the
overall structure of IR and general information about a specific class of
instructions or intrinsics.
