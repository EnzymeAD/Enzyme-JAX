#!/bin/bash
# Remove the build trees that cancelled jobs left in other runner slots of this
# project. ci/cscs-mi300.yml starts it in the background at the start of every job.
#
# A cancel kills the job's Slurm allocation, so neither after_script nor the
# runner's own cleanup runs, and the slot keeps ~700k files until another job of
# this project lands in it. CSCS cancels a PR's previous pipeline about a minute
# after every push starts a new one, so the new job is the natural place to clean
# up after the old one -- but only once that job is gone, hence REPEAT_MIN.
# cscs-mi300.md: "Job directory cleanup" has the details.
#
# A slot is swept only if all of these hold:
#   - it is <slots root>/<slot>/<this project> and not this job's own slot;
#   - its newest stage script names a CI job with no Slurm job (ci-<id>) left;
#   - nothing at its top level changed in the last STALE_MIN minutes.
# Only the job-generated dirs go; the checkout stays for the runner.
#
# Optional env:
#   STALE_MIN       minutes a slot must be untouched before it is swept (default 5)
#   REPEAT_MIN      keep making passes for this many minutes (default 0: one pass)
#   REPEAT_EVERY_S  seconds between passes (default 120)
#   DRY_RUN=1       report what would be removed (and every skip), remove nothing
set -uo pipefail   # no -e: a sweep problem must never fail the CI job

: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set}"
STALE_MIN="${STALE_MIN:-5}"
REPEAT_MIN="${REPEAT_MIN:-0}"
REPEAT_EVERY_S="${REPEAT_EVERY_S:-120}"
DRY_RUN="${DRY_RUN:-0}"
[[ "${STALE_MIN}" =~ ^[0-9]+$ ]] || STALE_MIN=5
[[ "${REPEAT_MIN}" =~ ^[0-9]+$ ]] || REPEAT_MIN=0
[[ "${REPEAT_EVERY_S}" =~ ^[1-9][0-9]*$ ]] || REPEAT_EVERY_S=120
BUILD_DIRS=(.bazel .julia .bazelisk .rocm Reactant.jl GB-25 bin run_julia.sh)

project="$(basename "${CI_PROJECT_DIR}")"
slots_root="$(dirname "$(dirname "${CI_PROJECT_DIR}")")"
me="$(id -un)"

log() { echo "sweep $(date +%H:%M:%S): $*"; }

# Expect $SCRATCH/gitlab-runner/f7t/<slot>/<project>; touch nothing on any other layout.
if [[ "${slots_root}" != */gitlab-runner/f7t ]]; then
  log "unexpected job dir layout (${CI_PROJECT_DIR}), skipping"
  exit 0
fi

sweep_slot() {
  local dir="$1" p
  for p in "${BUILD_DIRS[@]}"; do
    [[ -e "${dir}/${p}" || -L "${dir}/${p}" ]] || continue
    # "\;" not "+": Bazel's mode-000 sandbox dir must be opened up before find
    # descends into it. See cscs-mi300.md: "Job directory cleanup".
    find "${dir}/${p}" -type d ! -perm -u=rwx -exec chmod u+rwx {} \; 2>/dev/null
    rm -rf "${dir:?}/${p:?}" || log "${dir}/${p} not fully removed"
  done
  log "${dir}: done"
}

swept=0
sweep_pass() {
  local dir p newest_script owner
  # Without squeue there is no telling which slots are in use.
  if ! squeue -h -u "${me}" > /dev/null; then
    log "squeue failed, skipping this pass"
    return
  fi
  for dir in "${slots_root}"/*/"${project}"; do
    [[ -d "${dir}" && "${dir}" != "${CI_PROJECT_DIR}" ]] || continue
    local left=()
    for p in "${BUILD_DIRS[@]}"; do
      [[ -e "${dir}/${p}" || -L "${dir}/${p}" ]] && left+=("${p}")
    done
    (( ${#left[@]} )) || continue

    newest_script="$(ls -t "${dir}"/script_* 2>/dev/null | head -n 1)"
    owner="$(grep -oE 'CI_JOB_ID=[0-9]+' "${newest_script:-/dev/null}" 2>/dev/null | head -n 1 | cut -d= -f2)"
    if [[ -z "${owner}" ]]; then
      [[ "${DRY_RUN}" == 1 ]] && log "${dir}: owning job unknown, skipping"
      continue
    fi
    if [[ -n "$(squeue -h -u "${me}" -n "ci-${owner}" 2>/dev/null)" ]]; then
      [[ "${DRY_RUN}" == 1 ]] && log "${dir}: job ${owner} still queued or running, skipping"
      continue
    fi
    # Also skips a slot this or another sweeper is still deleting.
    if [[ -n "$(find "${dir}" -maxdepth 1 -mmin -"${STALE_MIN}" -print -quit 2>/dev/null)" ]]; then
      [[ "${DRY_RUN}" == 1 ]] && log "${dir}: changed in the last ${STALE_MIN} min, skipping"
      continue
    fi

    log "${dir}: job ${owner} is gone, removing ${left[*]}"
    swept=$((swept + 1))
    if [[ "${DRY_RUN}" != 1 ]]; then
      sweep_slot "${dir}" &   # slots in parallel; a 700k-file .bazel takes ~5 min
    fi
  done
  wait
}

end=$((SECONDS + REPEAT_MIN * 60))
while :; do
  sweep_pass
  (( SECONDS < end )) || break
  sleep "${REPEAT_EVERY_S}"
done
log "finished, ${swept} stale slot(s) swept"
exit 0
