#!/bin/bash
# Run Julia against the local build on an mi300 node, with the curated ROCm libs.
#
#   source ci/local/env.sh
#   ci/local/use-build.sh --project=$GB25_DIR -e 'using Reactant; @show Reactant.devices()'
#   ci/local/use-build.sh --project=$GB25_DIR -i            # interactive REPL
#
# Inside an existing allocation it uses it; otherwise it allocates one node itself.
# Debug knobs (AMD_SERIALIZE_KERNEL, AMD_LOG_LEVEL, HSA_XNACK, ...) are inherited
# via --export=ALL, so `AMD_LOG_LEVEL=3 ci/local/use-build.sh ...` just works.
set -euo pipefail
: "${BUILD_ROOT:?source ci/local/env.sh first}"
[[ -f "${BUILD_ROOT}/manifest.txt" ]] || echo "WARNING: no manifest.txt — has the build finished?" >&2

ALLOC=()
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  ALLOC=(--account="${MI300_ACCOUNT}" --partition="${MI300_PARTITION}"
         --nodes=1 --gpus-per-task=4 --time="${MI300_TIME:-01:00:00}")
fi
exec srun -n 1 --export=ALL "${ALLOC[@]}" --uenv "${UENV}" --view="${UENV_VIEW}" \
     --pty "$(dirname "$0")/julia.sh" "$@"
