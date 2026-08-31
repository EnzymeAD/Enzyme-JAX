#!/bin/bash
# 0.8 — gate the build before GB-25 is ever involved: instantiate a minimal env
# against the freshly built libReactantExtra.so, then run ci/local/check-build.jl.
# A failure here is a build problem; a GB-25 failure after this passes is not.
#
#   source ci/local/env.sh && sbatch ci/local/gate.sbatch
#
#SBATCH --job-name=mi300-gate
# Defaults only — Slurm cannot expand variables in #SBATCH directives.
# ci/local/submit.sh overrides account/partition/output from env.sh.
#SBATCH --account=a-c44
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/lraess/mi300-dev/logs/gate-%j.out
#SBATCH --error=/capstor/scratch/cscs/lraess/mi300-dev/logs/gate-%j.out
set -euo pipefail
MI300_DIR="${MI300_DIR:-${SLURM_SUBMIT_DIR}/ci/local}"
source "${MI300_ENV:-${MI300_DIR}/env.sh}"
GATE="${BUILD_ROOT}/gate-env"
mkdir -p "${GATE}"; cp "${REACTANT_DIR}/LocalPreferences.toml" "${GATE}/"

echo "=== gate job ${SLURM_JOB_ID} on $(hostname) — $(date -Is) ==="
set +e
srun -n1 --export=ALL --uenv "${UENV}" --view="${UENV_VIEW}" \
  "${MI300_DIR}/julia.sh" --project="${GATE}" --color=yes \
  "${BUILD_ROOT}/gate-setup.jl"
RC=$?
if [[ ${RC} -eq 0 ]]; then
  srun -n1 --export=ALL --uenv "${UENV}" --view="${UENV_VIEW}" \
    "${MI300_DIR}/julia.sh" --project="${GATE}" --color=yes \
    "${MI300_DIR}/check-build.jl"
  RC=$?
fi
set -e
[[ ${RC} -eq 0 ]] && RESULT="OK (job ${SLURM_JOB_ID})" || RESULT="FAILED (job ${SLURM_JOB_ID}, rc=${RC}) — see logs/gate-${SLURM_JOB_ID}.out"
sed -i "s|^- \*\*gate\*\*.*|- **gate**: ${RESULT}|" "${BUILD_ROOT}/STATUS.md" 2>/dev/null || true
echo "=== gate ${RESULT} ==="
exit ${RC}
