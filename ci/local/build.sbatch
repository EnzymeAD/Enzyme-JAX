#!/bin/bash
# Batch build of libReactantExtra.so on an mi300 compute node.
#
# Deliberately sbatch, not salloc: the build takes ~48 min and must survive the
# submitting shell going away. Watch it with `squeue -u $USER` and
# `tail -f $BUILD_ROOT/logs/build-<jobid>.out`; nothing is lost by disconnecting.
#
#   source ci/local/env.sh && sbatch ci/local/build.sbatch
#
#SBATCH --job-name=mi300-reactant-build
# Defaults only — Slurm cannot expand variables in #SBATCH directives.
# ci/local/submit.sh overrides account/partition/output from env.sh.
#SBATCH --account=a-c44
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/lraess/mi300-dev/logs/build-%j.out
#SBATCH --error=/capstor/scratch/cscs/lraess/mi300-dev/logs/build-%j.out
set -euo pipefail

MI300_DIR="${MI300_DIR:-${SLURM_SUBMIT_DIR}/ci/local}"
source "${MI300_ENV:-${MI300_DIR}/env.sh}"
STATUS="${BUILD_ROOT}/STATUS.md"

echo "=== mi300 build job ${SLURM_JOB_ID} on $(hostname) — $(date -Is) ==="
echo "  enzyme-jax  ${ENZYMEXLA_COMMIT}"
echo "  reactant    ${REACTANT_COMMIT}"
echo "  uenv        ${UENV}"
echo "  build root  ${BUILD_ROOT}"
echo "  disk cache  ${BAZEL_DISK_CACHE}"

sed -i "s|^- \*\*build\*\*.*|- **build**: RUNNING (job ${SLURM_JOB_ID}, started $(date -Is))|" "${STATUS}" 2>/dev/null || true

START=$(date +%s)
set +e
srun -n 1 --export=ALL --uenv "${UENV}" --view="${UENV_VIEW}" \
     "${MI300_DIR}/build.sh"
RC=$?
set -e
ELAPSED=$(( ($(date +%s) - START) / 60 ))

SO="${REACTANT_DIR}/deps/ReactantExtra/bazel-bin/libReactantExtra.so"
[[ -f "${SO}" ]] || SO=$(find "${REACTANT_DIR}" -name libReactantExtra.so -not -path '*/.git/*' 2>/dev/null | head -1)

if [[ ${RC} -eq 0 && -n "${SO}" && -f "${SO}" ]]; then
  RESULT="OK (job ${SLURM_JOB_ID}, ${ELAPSED} min, $(du -h "$(readlink -f "${SO}")" | cut -f1))"
  # Manifest: the only record of what this .so actually is.
  { echo "built_at        $(date -Is)"
    echo "job_id          ${SLURM_JOB_ID}"
    echo "node            $(hostname)"
    echo "elapsed_min     ${ELAPSED}"
    echo "uenv            ${UENV} (view=${UENV_VIEW})"
    echo "enzyme_jax      ${ENZYMEXLA_COMMIT}"
    echo "reactant        ${REACTANT_COMMIT}"
    echo "gb25            ${GB25_COMMIT}"
    echo "jax             ${JAX_COMMIT}"
    echo "xla             ${XLA_COMMIT}"
    echo "llvm            ${LLVM_COMMIT}"
    echo "triton          ${TRITON_COMMIT}"
    echo "julia           $(readlink -f "${JULIA}")"
    echo "target          gfx942 (MI300A, sramecc+:xnack-)"
    echo "library         $(readlink -f "${SO}")"
    echo "library_bytes   $(stat -c %s "$(readlink -f "${SO}")")"
    echo "enzymexla_local ${ENZYMEXLA_LOCAL:-<none, pinned tarball>}"
  } > "${BUILD_ROOT}/manifest.txt"
  echo "=== manifest ==="; cat "${BUILD_ROOT}/manifest.txt"
else
  RESULT="FAILED (job ${SLURM_JOB_ID}, rc=${RC}, ${ELAPSED} min) — see logs/build-${SLURM_JOB_ID}.out"
fi

sed -i "s|^- \*\*build\*\*.*|- **build**: ${RESULT}|" "${STATUS}" 2>/dev/null || true
echo "=== ${RESULT} ==="
exit ${RC}
