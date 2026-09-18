# `llvm-to-omp` regression tests

These tests cover the `--llvm-to-omp` pass, which lifts the LLVM-dialect
OpenMP *runtime* calls that clang emits (`__kmpc_fork_call`,
`__kmpc_for_static_init_*`, `__kmpc_dispatch_*`, `__kmpc_reduce_nowait`, …)
back into structured OpenMP dialect operations.

## Where the inputs come from

Each `.mlir` input is machine-generated, not hand-written, so that it has the
shape the pass actually sees in production rather than the shape raw clang
output happens to have. The distinction matters: `llvm-to-omp` runs *late*,
after the CFG has been lifted to SCF and after the pointer arithmetic has been
raised to `memref`. Run against raw `mlir-translate --import-llvm` output the
pass recognises only `__kmpc_fork_call`; the worksharing-loop, `single` and
reduction patterns all expect SCF structure that is not there yet.

The inputs were produced from small OpenMP C programs, one per construct:

```sh
# 1. clang -> LLVM IR -> LLVM dialect
clang -fopenmp -O1 -S -emit-llvm -o t.ll t.c
mlir-translate --import-llvm t.ll -o t.mlir

# 2. the pipeline prefix that runs before llvm-to-omp in practice
#    (taken from spmd-verify.sh, minus the symbol-privatize/symbol-dce
#    bookkeeping that would delete a test function with no `main` caller)
enzymexlamlir-opt --pass-pipeline='builtin.module(
    sroa-wrappers{set_private=false attributor=false},
    libdevice-funcs-raise,canonicalize,
    llvm-to-memref-access,polygeist-mem2reg,canonicalize,
    convert-llvm-to-cf,canonicalize,polygeist-mem2reg,canonicalize,
    enzyme-lift-cf-to-scf,canonicalize,
    func.func(canonicalize-loops),llvm.func(canonicalize-loops),
    canonicalize-scf-for,canonicalize,
    func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize,
    llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,
    simplify-affine-exprs,llvm-to-affine-access,canonicalize,
    func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,
    lower-affine,parallel-serialization,canonicalize)' t.mlir
```

Target/ABI noise from step 1 (TBAA and alias-scope metadata, `target_features`,
`passthrough`, `target_cpu`/`tune_cpu`, `uwtable_kind`, `dlti.dl_spec`,
parameter ABI attributes and the `no_unwind`/`convergent` call attributes) was
stripped so the inputs stay readable. `llvm.intr.lifetime.*` was dropped too:
it carries nothing for OpenMP raising.

The C sources behind these, and larger end-to-end reproducers, live in the
data-race-detection benchmark suite used by the SPMD verification project
(`llvmToOmpTests/` and `threadSanity*/`), which is where this pass was
originally developed and exercised. `masked_filter.mlir` and
`single_nowait.mlir` cover the two clause spellings that suite exercises and
the rest of these tests did not.

## Coverage

| test | construct |
|---|---|
| `parallel.mlir` | `parallel` |
| `num_threads.mlir` | `num_threads` |
| `wsloop_static.mlir` | `for` with the static schedule |
| `wsloop_dynamic.mlir` | `for` with a dynamic schedule |
| `wsloop_guided.mlir` | `schedule(guided, N)` |
| `wsloop_i64.mlir` | `for` over a 64-bit induction variable |
| `ordered.mlir` | `ordered` |
| `sections.mlir` | `sections` |
| `barrier.mlir` | `barrier` |
| `single_master.mlir` | `single`, `master` |
| `single_nowait.mlir` | `single nowait` |
| `masked.mlir` | `masked` |
| `masked_filter.mlir` | `masked filter(N)` |
| `critical.mlir` | `critical` |
| `flush_taskyield.mlir` | `flush`, `taskyield` |
| `cancel.mlir` | `cancel`, `cancellation point` |
| `task.mlir` | `task`, `taskwait` |
| `taskloop.mlir` | `taskloop grainsize(N)`, `taskgroup` |
| `taskloop_num_tasks.mlir` | `taskloop num_tasks(N)` |
| `teams.mlir` | `teams num_teams(N)` |
| `reduction.mlir` | `reduction` |

## What each test asserts

Every test runs `FileCheck` with `--implicit-check-not="llvm.call @__kmpc"`,
so a test fails if *any* OpenMP runtime call survives anywhere in the module —
not merely if the expected `omp.*` ops are missing. Forward declarations of the
runtime functions are expected to remain; `symbol-dce` removes them downstream.

The dead outlined function (`@<name>.omp_outlined`) is likewise still present
after the pass: its body has been inlined into the `omp.parallel`, and the
production pipeline runs `symbol-dce` immediately afterwards to drop it.

`cancel.mlir` is the one test that runs `--symbol-dce` itself. `omp.cancel` and
`omp.cancellation_point` verify only inside the construct they cancel, so the
pass leaves the runtime calls alone in that dead copy, where there is no
enclosing construct — running the DCE the pipeline would run anyway removes the
copy and with it the calls.
