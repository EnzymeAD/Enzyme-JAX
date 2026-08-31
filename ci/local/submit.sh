#!/bin/bash
# Submit a ci/local batch job with account, partition and log path taken from env.sh.
#
# Slurm does not expand variables in #SBATCH directives, so the literals inside the
# .sbatch files can only ever be defaults. Command-line flags override them, which is
# what this wrapper supplies — so overriding MI300_ACCOUNT / BUILD_ROOT in the
# environment actually takes effect.
#
#   source ci/local/env.sh && ci/local/submit.sh build.sbatch
#   MI300_ACCOUNT=a-xyz ci/local/submit.sh gate.sbatch --time=01:00:00
set -euo pipefail
: "${BUILD_ROOT:?source ci/local/env.sh first}"
JOB="${1:?usage: submit.sh <build.sbatch|gate.sbatch> [extra sbatch args...]}"; shift
DIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f "${DIR}/${JOB}" ]] || { echo "ERROR: no such job script: ${DIR}/${JOB}" >&2; exit 1; }
NAME="$(basename "${JOB}" .sbatch)"
mkdir -p "${BUILD_ROOT}/logs"
exec sbatch \
  --account="${MI300_ACCOUNT}" \
  --partition="${MI300_PARTITION}" \
  --output="${BUILD_ROOT}/logs/${NAME}-%j.out" \
  --error="${BUILD_ROOT}/logs/${NAME}-%j.out" \
  --export="ALL,MI300_ENV=${DIR}/env.sh,MI300_DIR=${DIR}" \
  "$@" "${DIR}/${JOB}"
