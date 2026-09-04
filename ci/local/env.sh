#!/bin/bash
# Pinned configuration for the local MI300 build tree. Source this, don't run it.
#
# Everything downstream (build, run, bisect) is measured against the tuple below.
# While characterising a crash these MUST stay fixed: two runs against a floating
# `main` cannot distinguish a real regression from upstream drift.
#
# Every value is `${VAR:-default}`, so anything can be overridden from the
# environment without editing this file. Export first, then source -- bash reverts
# temporary assignments made on a `source` command, so the inline form silently
# does nothing:
#   export MI300_ACCOUNT=a-xyz REACTANT_COMMIT=<sha>
#   source ci/local/env.sh
# The defaults are the CSCS beverin / lraess setup.

# ---- machine ----
export SCRATCH="${SCRATCH:-/capstor/scratch/cscs/lraess}"   # not reliably set inside srun
export BUILD_ROOT="${BUILD_ROOT:-${SCRATCH}/mi300-dev}"
export UENV="${UENV:-56ebb909a6164680}"   # prgenv-gnu/7.2.3:2753912524, ROCm 7.2.3, beverin
export UENV_VIEW="${UENV_VIEW:-default}"
export JULIA="${JULIA:-/users/lraess/julia_amd/julia_amd-1.11}"
# Deliberately MI300_* and not SLURM_*: SLURM_PARTITION and SLURM_ACCOUNT are read
# by srun/sbatch as implicit --partition/--account, so exporting them would silently
# steer every unrelated Slurm command in the same shell.
export MI300_ACCOUNT="${MI300_ACCOUNT:-a-c44}"
export MI300_PARTITION="${MI300_PARTITION:-mi300}"

# ---- pins (resolved 2026-08-31) ----
# Enzyme-JAX: origin/lr/up-mi300. HEAD (8296190a) is CI-only (ci/*.yml, ci/*.md) and
# produces a byte-identical library, so the build does not need it pushed.
export ENZYMEXLA_COMMIT="${ENZYMEXLA_COMMIT:-81775cb841552f26eed78737670f8b2103704a58}"
export REACTANT_COMMIT="${REACTANT_COMMIT:-def6fd8e9cd4a44d147b323e7807ba47bc85bb78}"
export GB25_COMMIT="${GB25_COMMIT:-60785115ed4a70b5bc31b6714b38c71a1dc726de}"
# Toolchain, resolved through Enzyme-JAX -> JAX -> XLA -> {LLVM,Triton}. Informational:
# these follow from ENZYMEXLA_COMMIT, recorded so a manifest can be checked by eye.
export JAX_COMMIT="${JAX_COMMIT:-cec06d116c05f0d52adfceee3d3b730fdbcb0ce5}"
export XLA_COMMIT="${XLA_COMMIT:-753e5e56a0e465dbf7ee166573122564d83a1746}"
export LLVM_COMMIT="${LLVM_COMMIT:-084f6484d76625b995c07f512a323ab7cdc5351d}"
export TRITON_COMMIT="${TRITON_COMMIT:-72259b1cc3c543c361dcd185a6ff89662e8ed52f}"

# ---- derived paths ----
export REACTANT_DIR="${REACTANT_DIR:-${BUILD_ROOT}/Reactant.jl}"
export GB25_DIR="${GB25_DIR:-${BUILD_ROOT}/GB-25}"
export LOCAL_BIN="${LOCAL_BIN:-${BUILD_ROOT}/bin}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-${BUILD_ROOT}/.julia}"
export BAZELISK_HOME="${BAZELISK_HOME:-${BUILD_ROOT}/.bazelisk}"
export BAZEL_OUTPUT_ROOT="${BAZEL_OUTPUT_ROOT:-${BUILD_ROOT}/.bazel}"
export BAZEL_DISK_CACHE="${BAZEL_DISK_CACHE:-${BUILD_ROOT}/.bazel-disk}"
export ROCM_PATH="${ROCM_PATH:-/user-environment/env/default}"
export JULIA_PKG_SERVER_REGISTRY_PREFERENCE=conservative
export JULIA_MAX_NUM_PRECOMPILE_FILES=3

# The bazel/bazelisk shims must win over anything else on PATH. Idempotent: sourcing
# this twice must not stack duplicate entries.
case ":${PATH}:" in *":${LOCAL_BIN}:"*) ;; *) export PATH="${LOCAL_BIN}:${PATH}" ;; esac

# CI compatibility: the extracted build.sh guards on BUILD_ROOT, but the shims
# (shared verbatim with CI) still guard on CI_PROJECT_DIR.
export CI_PROJECT_DIR="${BUILD_ROOT}"

uenv_run() { uenv run "${UENV}" --view="${UENV_VIEW}" -- "$@"; }
