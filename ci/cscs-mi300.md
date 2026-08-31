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
- `build_reactant.sh` — builds the overlay + runs the Bazel build.
- `run_julia.sh` — launches Julia with `LD_LIBRARY_PATH` pointed at the curated
  ROCm libs.

### Heredoc levels

Most of the yml's apparent line noise is heredoc escaping. There are three levels,
and a `$` must be escaped once per level it should *survive*:

| Level | What runs it | Escape to defer past it |
|---|---|---|
| 0 — the yml | the runner's shell writes `build_reactant.sh` (unquoted `<< BUILDSCRIPT`) | `\$` |
| 1 — `build_reactant.sh` | runs inside `srun`, writes the clang wrappers (unquoted `<< WRAPPER`) | `\\\$` |
| 2 — the clang wrapper | runs per compiler invocation | — |

So `${CI_PROJECT_DIR}` (bare) is baked in when the yml step runs;
`\${OVERLAY_RESOURCE_DIR}` resolves inside `srun`; and `\\\$@` reaches the wrapper
as a literal `$@`. `run_julia.sh` uses a **quoted** heredoc (`<< 'RUNSCRIPT'`), so
nothing expands at write time and every var resolves at run time — that is why its
body reads like ordinary shell.

### Script shorthands

All `script:` steps share one shell, so `SRUN_UENV` and `JULIA_FLAGS` are defined
once as plain shell variables near the top and reused unquoted (deliberately — they
must word-split into separate argv entries). They hold no paths with spaces. This is
the same mechanism the `export PATH=` step already relies on.

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

### Bazel output root

Bazel and bazelisk default to `~/.cache/`, which on Alps is a small, shared home
directory — a full XLA build fills it and the failure is confusing. `BAZELISK_HOME`
and `JULIA_DEPOT_PATH` are redirected under `CI_PROJECT_DIR` by job variables, but
Bazel's own output base is only settable per-invocation, and Reactant's
`build_local.jl` calls `bazelisk` **by name** rather than through a configurable
path. So the job writes `bin/bazelisk` and `bin/bazel` shims that inject
`--output_user_root="${BAZEL_OUTPUT_ROOT}"` and `exec` the real binary, and prepends
`bin/` to `PATH`. Both names are shimmed because different call sites use each.

The shims **refuse to run** if `BAZEL_OUTPUT_ROOT` is unset or points outside
`CI_PROJECT_DIR`, so a mistake fails loudly instead of silently filling `$HOME`. A
separate step then asserts `command -v bazelisk` resolves to the shim, catching a
mis-ordered `PATH` before the ~25 min build rather than after.

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
device-bitcode step also defensively replaces a symlinked parent with a real dir —
removing the *link* never touches its target in the read-only view.

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

### hipcc resource dir

Enzyme-JAX's own `XLA_PATCHES` (in [`workspace.bzl`](../workspace.bzl), added
2026-08-10) declare the hipcc resource directory to the ROCm crosstool at a
**hardcoded** path:

```
${ROCM_PATH}/lib/llvm/lib/clang/22/include
```

and pass it twice — once as a package token and once as
`repository_ctx.path(...).realpath`. Because `ROCM_PATH` is set, `rocm_configure`
symlinks the overlay to `rocm/rocm_dist`, so that resolves into the overlay. The
`.realpath` call **fails outright if the path does not exist**, and the whole
`local_config_rocm` fetch dies before a single target is built:

```
ERROR: ... An error occurred during the fetch of repository 'local_config_rocm':
Error: ${ROCM_PATH}/lib/llvm/lib (No such file or directory)
ERROR: no such package '@@local_config_rocm//rocm': ...
```

The overlay's copied resource dir lives at `${ROCM_PATH}/llvm/lib/clang/<v>` (that
is what the clang wrapper reports), which is *not* the same path — note the extra
`lib/`. So the clang-wrapper step also aliases it:

```
${ROCM_PATH}/lib/llvm/lib/clang/{<v>,22} -> ${ROCM_PATH}/llvm/lib/clang/<v>
```

Aliasing (rather than a second copy) is deliberate: the declared `realpath` then
resolves to exactly the dir the wrapper forces with `-resource-dir`, so what the
crosstool declares and what compiles actually include agree. Both `<v>` and the
literal `22` are linked so the alias survives a UENV whose clang is not 22 (the
patch's `22` stays hardcoded either way). `lib/llvm/` already exists as a real dir
from the device-bitcode step, so only `lib/llvm/lib/clang` is created here.

The overlay check asserts `lib/llvm/lib/clang/22/include/stddef.h`.

### Overlay lib/ layout

`lib/` is the one directory built from **per-file** symlinks rather than a
whole-directory link. `rocm_configure` references individual libraries as
`rocm_dist/lib/<name>`, but the UENV splits them across the view's `lib/` and
`lib64/`, and several ROCm libraries live only in their own spack package. So the
overlay seeds `lib/` from the view's `lib/` then `lib64/`, then auto-fills anything
still missing by scanning every spack package's `lib/` and `lib64/`. First writer
wins (`[[ -e ]] ||`), so the view's copy always takes precedence over a spack one.

`lib64/` and `share/` *are* whole-directory symlinks — nothing needs to write into
them. `amdgcn/`, `bin/`, and `lib/llvm/` must stay real directories; see *amdgcn
must stay a real dir* and *clang wrapper + HIP_CLANG_PATH*.

### Bazel extraopts

`build_local.jl --extraopt=...` is how overlay/toolchain settings reach
`rocm_configure` without editing Reactant's checked-in `.bazelrc`:

- `--repo_env=ROCM_PATH` points the whole ROCm autoconfiguration at the overlay.
- `--repo_env=TF_ROCM_AMDGPU_TARGETS=gfx942` pins the target arch *and* suppresses
  `rocm_agent_enumerator`, which would otherwise need a GPU-attached `srun`.
- `--action_env=PATH=${PATH}` forwards the build shell's `PATH`, which already
  leads with the wrapper dir (exported a few lines above). Do **not** re-prepend the
  wrapper dir here — it only duplicates the entry.
- `--action_env=HIP_CLANG_PATH` / `CLANG_COMPILER_PATH` / `CLANGXX_COMPILER_PATH`
  route every compiler entry point through the wrappers.
- `--linkopt`/`--host_linkopt=-fuse-ld=lld` — the UENV has no `ld.gold`.

### Curated runtime libs

`libReactantExtra.so` dlopens ROCm libs (`librocblas.so.5`, …) at Julia load time.
Putting the whole overlay `lib/` (or the view lib dir) on `LD_LIBRARY_PATH` also
exposes the UENV's **old libcurl / libstdc++ / libLLVM**, which shadow Julia's
bundled copies and break `Pkg` downloads (`curl_easy_setopt: 48`). So the build
curates `.rocm/reactant_libs`: every overlay ROCm `.so` **except** any soname Julia
also bundles (matched against Julia's `lib/julia`). That match is derived from
`${JULIA}` via `readlink -f` — `JULIA` may be a *symlink* to the real
`<prefix>/bin/julia`, and without resolving it the dirname chain walks the wrong
prefix, the glob matches nothing, and every shadowing lib is curated anyway (the
failure is silent at build time and only surfaces later as `curl_easy_setopt: 48`). Excluded libs that ROCm itself
needs still resolve via each ROCm lib's own (spack) RPATH.

### run_julia.sh

Every Reactant-loading Julia step (precompile + all three sims) runs through
`run_julia.sh` **inside `srun --uenv`** (so `/user-environment`, the overlay
symlink targets, is mounted). It sets `LD_LIBRARY_PATH` to the curated dir
**only** — deliberately not appending the inherited view `LD_LIBRARY_PATH`, which
still carries the shadowing libcurl. The bare-`${JULIA}` steps (lines that
instantiate/build Reactant itself) don't load `libReactantExtra.so` and correctly
keep their own `--project` without the wrapper.

### Collectives gate

`sharded_baroclinic_instability_simulation_compile.jl` writes its
`optimised_sharded_baroclinic_instability_*.xla` files **relative to the current
working directory**, and the threshold check greps for them under `GB25_DIR`. All
`script:` steps share one shell, so the `cd "${REACTANT_DIR}/deps"` from the
Reactant instantiate step is still in effect when the compile step runs — the files
landed in `Reactant.jl/deps`, `find` matched nothing, every `NUM_COLLECTIVES` was
0, and the gate passed **without checking anything**. It had been green for that
reason, not because the collective counts were acceptable.

Two changes: the compile step now `cd`s into `GB25_DIR` first, and the gate fails
loudly when it finds no `.xla` files instead of silently comparing zeroes. The
thresholds themselves are still the A100 numbers and still need calibrating — the
first run that actually reports counts is the one to calibrate from.

### Collective timeouts

XLA warns when a thread has waited on the first collective rendezvous longer than
`--xla_gpu_first_collective_call_warn_stuck_timeout_seconds` (10 s by default) and
kills the job past the matching `..._terminate_timeout_seconds`. Clique init for
`devices=4:[0,1,2,3]` measured **~57 s** on this node, so every run logged three
alarming "may be stuck / leader can be deadlocked" errors and then retracted them
("Warning above was a false-positive"). The job sets `XLA_FLAGS` to raise the warn
threshold to 120 s. Note `GordonBell25.preamble()` refuses to set these itself and
directs you to `XLA_FLAGS`, which is why they live in the job variables.

If clique init ever *does* hang, that same knob is where to lower the terminate
timeout to get a fast failure.

## UENV bump checklist

When `UENV:` (and thus the ROCm/LLVM toolchain) changes, re-verify:

1. **Overlay check** output (printed each build): headers `OK`, both `amdgcn/bitcode`
   and `lib/llvm/amdgcn/bitcode` `OK`, and `curated N runtime libs` with N > 0.
2. The **LLVM-header shadow guard** still passes (no `include/{llvm,mlir,clang}`).
3. `clang -print-resource-dir` version dir still resolves (the wrapper cp's it),
   and `lib/llvm/lib/clang/22/include` still aliases onto it — see *hipcc resource
   dir* if the clang major version moved off 22.
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
