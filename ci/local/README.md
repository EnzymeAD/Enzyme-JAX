# Local MI300 build

Build and test `libReactantExtra.so` for MI300A (gfx942) on CSCS beverin **without
going through CI**. A CI round trip is ~81 minutes, ~48 of which is a full rebuild;
this gets you a REPL against the same library instead.

The build logic in [`build.sh`](build.sh) is the *same script CI runs* — see
[Relationship to CI](#relationship-to-ci). Every workaround in it is documented in
[`../cscs-mi300.md`](../cscs-mi300.md); that runbook remains the reference.

## Quickstart

```bash
cd <Enzyme-JAX checkout>
source ci/local/env.sh          # pins + paths; safe to source repeatedly
ci/local/setup.sh               # login node: shims, pinned checkouts, Julia depot
ci/local/submit.sh build.sbatch # compute node: ~31 min cold, much less warm
ci/local/submit.sh gate.sbatch  # prove the library before involving GB-25
```

Then iterate:

```bash
ci/local/use-build.sh --project=$BUILD_ROOT/gate-env -i    # interactive REPL
AMD_LOG_LEVEL=3 AMD_SERIALIZE_KERNEL=3 \
  ci/local/use-build.sh --project=$GB25_DIR script.jl      # debug knobs pass through
```

## Files

| File | Runs on | What it does |
|---|---|---|
| `env.sh` | anywhere | Pins and paths. **Source it, don't run it.** |
| `setup.sh` | login node | Bazel shims, pinned Reactant/GB-25 checkouts, `ENZYMEXLA_COMMIT` rewrite, Julia depot. Idempotent. |
| `build.sh` | inside uenv | Assembles the ROCm overlay at `$BUILD_ROOT/.rocm`, then runs Bazel via Reactant's `build_local.jl`. Shared with CI. |
| `build.sbatch` | compute node | Batch wrapper around `build.sh`; writes `manifest.txt` on success. |
| `gate.sbatch` | compute node | Instantiates a minimal env, runs `check-build.jl`. |
| `check-build.jl` | compute node | Loads Reactant, asserts 4 agents, runs a plain and a 4-way sharded `@jit sum`. |
| `julia.sh` | inside uenv | Julia with `LD_LIBRARY_PATH` = curated ROCm libs **only**. |
| `use-build.sh` | anywhere | Runs Julia against the build on an mi300 node, allocating one if needed. |
| `submit.sh` | login node | Submits a `.sbatch` with account/partition/log path from `env.sh`. |

## Overriding anything

Every value in `env.sh` is `${VAR:-default}`. **Export first, then source** — bash
reverts temporary assignments made on a `source` command, so the inline form
silently does nothing:

```bash
export MI300_ACCOUNT=a-xyz REACTANT_COMMIT=<sha>
source ci/local/env.sh
```

Overriding `REACTANT_COMMIT` / `GB25_COMMIT` is how you bisect an upstream
regression without editing a tracked file.

Slurm cannot expand variables in `#SBATCH` directives, so the literals in the
`.sbatch` files are only fallbacks. `submit.sh` passes the real values as
command-line flags, which override them — that is why you should submit through it
rather than calling `sbatch` directly.

`MI300_ACCOUNT`/`MI300_PARTITION` are deliberately *not* named `SLURM_ACCOUNT`/
`SLURM_PARTITION`: those are read by `srun`/`sbatch` as implicit `--account`/
`--partition` and would silently steer every unrelated Slurm command in your shell.

## Building a local Enzyme-JAX tree

By default Bazel fetches Enzyme-JAX as a GitHub tarball at `ENZYMEXLA_COMMIT`, so
the commit must be **pushed**. To build uncommitted local changes instead:

```bash
export ENZYMEXLA_LOCAL=$SCRATCH/Enzyme-JAX
ci/local/submit.sh build.sbatch
```

`build.sh` copies that tree aside and replays the `http_archive`'s `patch_cmds` on
the copy, never on your working tree. **Untested as of 2026-08-31** — the default
pinned path is what has been exercised.

## Mutable state lives outside the repo

Nothing here writes into the checkout. Build state, logs and results live under
`$BUILD_ROOT` (default `$SCRATCH/mi300-dev`), which is deliberately not a git
directory — it changes on every job and would otherwise produce constant churn:

```
$BUILD_ROOT/STATUS.md      # phase/step status, machine-updated; read this first
$BUILD_ROOT/manifest.txt   # exactly what the current .so is (all pinned SHAs)
$BUILD_ROOT/logs/          # build-<jobid>.out, gate-<jobid>.out
$BUILD_ROOT/.bazel-disk/   # disk cache; makes rebuilds incremental
$BUILD_ROOT/gate-env/      # minimal Reactant env for quick experiments
```

Resuming from a cold shell:

```bash
source ci/local/env.sh
cat $BUILD_ROOT/STATUS.md
squeue -u $USER
tail -50 $BUILD_ROOT/logs/build-*.out
```

## Reproducibility

`manifest.txt` is the record of what a given `.so` actually is: Enzyme-JAX,
Reactant, GB-25, JAX, XLA, LLVM and Triton SHAs, the uenv id, the Julia binary and
the target. `env.sh` pins Reactant and GB-25 to **SHAs, not `main`** on purpose:
while characterising a crash, a floating upstream makes it impossible to separate a
real regression from drift, or to reproduce today's failure tomorrow.

## Relationship to CI

`ci/cscs-mi300.yml` generates its build script inline from a three-level heredoc.
`build.sh` is that script extracted to a real file, verified byte-identical to the
generated one (replay the yml's heredoc with sentinel values, round-trip, diff).

**The yml still generates its own copy today.** Pointing it at `build.sh` is a
separate change; until then the two are identical but independent, so a fix applied
to one must be applied to the other. Making CI call this file is what removes the
drift for good — and removes one heredoc level with it, along with the `\\\$`
escaping rules described in `../cscs-mi300.md`.
