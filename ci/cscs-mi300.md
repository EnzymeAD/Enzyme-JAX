# CSCS MI300 (gfx942) ROCm pipeline — maintainer runbook

Companion to [`cscs-mi300.yml`](cscs-mi300.yml). The yml keeps one-line "why"
comments; the full rationale for each non-obvious workaround lives here, keyed by
the section titles the yml points at (e.g. *see cscs-mi300.md: "LLVM headers"*).

## Overview

The job builds `libReactantExtra.so` for AMD MI300 with `--config=rocm`, then
compiles/runs/validates the GB-25 sharded baroclinic-instability simulation.

The CSCS baremetal runner executes `script:` steps on the compute node **without
the UENV mounted** — `/user-environment/` is only visible inside `srun`. So the
build runs a generated `build_reactant.sh` inside `srun --uenv`, which first
assembles a writable ROCm **overlay** at `${CI_PROJECT_DIR}/.rocm` (symlinks into
the UENV view + spack package store) that stands in for a normal `ROCM_PATH`, then
invokes Bazel against it.

Two generated scripts:
- `build_reactant.sh` — builds the overlay + runs the Bazel build (unquoted
  heredoc: `${CI_PROJECT_DIR}` etc. expand at *write* time; runtime shell vars use
  `\$`).
- `run_julia.sh` — launches Julia with `LD_LIBRARY_PATH` pointed at the curated
  ROCm libs (quoted heredoc: everything resolves at *run* time inside `srun`).

## Dependency pins

The compiled XLA/Triton/LLVM are **not** governed by Reactant `main` (the job only
clones it). CI rewrites Reactant's `ENZYMEXLA_COMMIT` to this repo's SHA, and the
chain is: **Enzyme-JAX → JAX (`JAX_COMMIT` in `workspace.bzl`) → XLA
(`revision.bzl`) → LLVM + Triton (XLA `third_party/{llvm,triton}/workspace.bzl`)**.
To pin the toolchain, bump `JAX_COMMIT`. Resolve the full chain with:

```
JAX=$(grep -oP 'JAX_COMMIT = "\K[0-9a-f]+' workspace.bzl)
XLA=$(curl -fsSL "https://raw.githubusercontent.com/jax-ml/jax/$JAX/third_party/xla/revision.bzl" | grep -oP 'XLA_COMMIT = "\K[0-9a-f]+')
curl -fsSL "https://raw.githubusercontent.com/openxla/xla/$XLA/third_party/llvm/workspace.bzl"   | grep LLVM_COMMIT
curl -fsSL "https://raw.githubusercontent.com/openxla/xla/$XLA/third_party/triton/workspace.bzl" | grep TRITON_COMMIT
```

## Workarounds

### LLVM headers

The overlay seeds `${ROCM_PATH}/include` from the UENV view. `rocm_configure` globs
that into `local_config_rocm`'s `rocm_headers_includes`, and Bazel places that `-I`
**ahead of** `-isystem external/llvm-project/...`. So any `llvm/`, `mlir/`, `clang/`
in the overlay **shadows the LLVM/MLIR that XLA is built against** (LLVM 23) with
the UENV's older copy — for ROCm-toolchain targets only. Symptoms (both were long
misdiagnosed as upstream Triton/MLIR bugs):

```
no member named 'getEmptyKey' in 'llvm::DenseMapInfo<xla::SymbolicExpr>'   (factory_rocm.cc)
no type named 'PropertyRef' in namespace 'mlir'                            (triton_rocm.cc)
```

Fix: the include-seeding loop skips `llvm llvm-c mlir mlir-c clang clang-c lld
lldb polly`. A hard guard (`exit 1` if `include/{llvm,mlir,clang}` reappears)
protects against regressions on a UENV bump. Only ROCm's own headers (`hip/`,
`rocblas/`, `miopen/`, …) belong in the overlay.

### rocm-smi header

`rocm-smi-lib` is a separate spack package not exposed in the view. Newer XLA
(`xla/stream_executor/rocm/rocm_smi_util.cc`) includes
`rocm/include/rocm_smi/rocm_smi.h`, so we seed `rocm_smi/` from its package
(well-known names first, then a header-gated scan of all spack packages).
MIOpen is handled the same way.

### amdgcn must stay a real dir

`.rocm/amdgcn` is intentionally **not** a whole-dir symlink into the view. It must
be a real, writable directory so the device-bitcode step can place a `bitcode`
symlink under it. Symlinking it to the read-only view makes
`${ROCM_PATH}/amdgcn/bitcode` read-only (`rm: ... Is a directory`). The
device-bitcode step also defensively replaces a symlinked parent with a real dir.

### Device bitcode (two locations)

ROCm device bitcode (`ocml.bc`, `ockl.bc`, …) must appear at **both**
`${ROCM_PATH}/amdgcn/bitcode` (HIP clang's default search path) and
`${ROCM_PATH}/lib/llvm/amdgcn/bitcode` (XLA's `generate_amdgpu_device_lib_data`
genrule references `rocm_dist/lib/llvm/amdgcn/bitcode/{ocml,ockl}.bc`). We resolve
the source dir (following an existing link, else a candidate scan of view + spack)
and symlink both.

### clang wrapper + HIP_CLANG_PATH

`rocm_configure.bzl` hardcodes `${ROCM_PATH}/llvm/bin/clang` for
`-print-resource-dir`. We wrap the UENV clang so its resource-dir is reported and
forced (`-resource-dir <overlay>`) inside the overlay; otherwise clang reports a
`/user-environment/.../lib/clang/<v>/include` path that Bazel treats as an
undeclared absolute include (breaks assembly compiles). The wrapper also maps
`-fuse-ld=gold` → `-fuse-ld=lld`.

`.rocm/bin` is a real dir (not a whole-dir symlink) so we can override
`bin/clang{,++}` — HIP compiles invoke `${ROCM_PATH}/bin/clang++` directly,
bypassing `PATH`. Additionally, HIP `.cu.cc` compiles go through **hipcc**, which
finds clang via `HIP_CLANG_PATH`; without pointing that at the wrapper dir, hipcc
reaches the view clang directly and leaks the undeclared resource-dir path again.
Hence `HIP_CLANG_PATH` is exported and passed as an `--action_env`.

### Curated runtime libs

`libReactantExtra.so` dlopens ROCm libs (`librocblas.so.5`, …) at Julia load time.
Putting the whole overlay `lib/` (or the view lib dir) on `LD_LIBRARY_PATH` also
exposes the UENV's **old libcurl / libstdc++ / libLLVM**, which shadow Julia's
bundled copies and break `Pkg` downloads (`curl_easy_setopt: 48`). So the build
curates `.rocm/reactant_libs`: every overlay ROCm `.so` **except** any soname Julia
also bundles (matched against Julia's `lib/julia`). Excluded libs that ROCm itself
needs still resolve via each ROCm lib's own (spack) RPATH.

### run_julia.sh

Every Reactant-loading Julia step (precompile + all three sims) runs through
`run_julia.sh` **inside `srun --uenv`** (so `/user-environment`, the overlay
symlink targets, is mounted). It sets `LD_LIBRARY_PATH` to the curated dir
**only** — deliberately not appending the inherited view `LD_LIBRARY_PATH`, which
still carries the shadowing libcurl. The bare-`${JULIA}` steps (lines that
instantiate/build Reactant itself) don't load `libReactantExtra.so` and correctly
keep their own `--project` without the wrapper.

## UENV bump checklist

When `UENV:` (and thus the ROCm/LLVM toolchain) changes, re-verify:

1. **Overlay check** output (printed each build): headers `OK`, both `amdgcn/bitcode`
   and `lib/llvm/amdgcn/bitcode` `OK`, and `curated N runtime libs` with N > 0.
2. The **LLVM-header shadow guard** still passes (no `include/{llvm,mlir,clang}`).
3. `clang -print-resource-dir` version dir still resolves (the wrapper cp's it).
4. spack package globs still match: `miopen-hip-*`, `rocm-smi-lib-*`,
   `rocm-device-libs-*` / `llvm-amdgpu-*` (bitcode).
5. `ROCM_PATH` view path (`/user-environment/env/default`) still valid.
6. The runtime soname the loader wants (currently `librocblas.so.5`) — if the ROCm
   major version changed, confirm the curated dir has the new soname.

## Diagnostics

The build prints a compact `=== overlay check ===` block: presence of key headers,
device bitcode (both locations), the shadow guard (fatal), and the curated lib
count. It's cheap and always on — it's the fastest signal that a UENV bump broke
an assumption before the ~25 min Bazel build wastes time.
