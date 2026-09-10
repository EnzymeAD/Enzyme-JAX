#!/bin/bash
# Submit a ci/local batch job with account, partition and log path taken from env.sh.
#
# Slurm does not expand variables in #SBATCH directives, so the literals inside the
# .sbatch files can only ever be defaults. Command-line flags override them, which is
# what this wrapper supplies — so overriding MI300_ACCOUNT / BUILD_ROOT in the
# environment actually takes effect.
#
#   source ci/local/env.sh && ci/local/submit.sh build.sbatch
#   ci/local/submit.sh $SCRATCH/mi300-reactant-debug/drivers/bisect.sbatch
#   MI300_ACCOUNT=a-xyz ci/local/submit.sh gate.sbatch --time=01:00:00
set -euo pipefail
: "${BUILD_ROOT:?source ci/local/env.sh first}"
JOB="${1:?usage: submit.sh <job.sbatch|path/to/job.sbatch> [extra sbatch args...]}"; shift
DIR="$(cd "$(dirname "$0")" && pwd)"
# A bare name resolves against this directory; a path is taken as given, so job scripts
# kept outside the repo (e.g. an investigation's own drivers/) can be submitted too.
if [[ -f "${JOB}" ]]; then
  JOB_PATH="$(cd "$(dirname "${JOB}")" && pwd)/$(basename "${JOB}")"
elif [[ -f "${DIR}/${JOB}" ]]; then
  JOB_PATH="${DIR}/${JOB}"
else
  echo "ERROR: no such job script: ${JOB} (also tried ${DIR}/${JOB})" >&2; exit 1
fi
NAME="$(basename "${JOB_PATH}" .sbatch)"
mkdir -p "${BUILD_ROOT}/logs"
exec sbatch \
  --account="${MI300_ACCOUNT}" \
  --partition="${MI300_PARTITION}" \
  --output="${BUILD_ROOT}/logs/${NAME}-%j.out" \
  --error="${BUILD_ROOT}/logs/${NAME}-%j.out" \
  --export="ALL,MI300_ENV=${DIR}/env.sh,MI300_DIR=${DIR}" \
  "$@" "${JOB_PATH}"
