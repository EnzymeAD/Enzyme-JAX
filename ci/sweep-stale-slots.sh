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
#   - its newest stage script names a CI job that squeue no longer lists, or lists
#     only as COMPLETING (its CI stages are over, and a job stuck there can stay
#     for an hour);
#   - nothing at its top level changed in the last STALE_MIN minutes.
# Each build dir is first claimed by renaming it to .swept.<sweeper>.<name>. The
# rename is atomic, so when several jobs sweep at once each dir goes to exactly one
# of them. A claim whose sweeper job is gone (killed mid-delete) is claimed again.
# Deletions run in the background, so a long one does not hold up later passes.
# Only the job-generated dirs go; the checkout stays for the runner.
#
# Manual use: CI_PROJECT_DIR=$SCRATCH/gitlab-runner/f7t/manual/<project-id> sweeps
# every slot of that project ("manual" is not a real slot, so none is excluded).
#
# Optional env:
#   STALE_MIN       minutes a slot must be untouched before it is swept (default 5)
#   REPEAT_MIN      keep making passes for this many minutes (default 0: one pass)
#   REPEAT_EVERY_S  seconds between passes (default 120)
#   DRY_RUN=1       report what would be claimed, change nothing
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
MANUAL_CLAIM_MIN=60   # a manual sweep's claims count as abandoned after this long

project="$(basename "${CI_PROJECT_DIR}")"
slots_root="$(dirname "$(dirname "${CI_PROJECT_DIR}")")"
me="$(id -un)"
sweeper="${CI_JOB_ID:-manual}"   # the <sweeper> in this run's claim names
[[ "${sweeper}" =~ ^[0-9]+$ ]] || sweeper=manual

log() { echo "sweep $(date +%H:%M:%S): $*"; }

# Log why a slot is skipped only when the reason changes, so a slot that waits
# through many passes says so once instead of every 2 min.
declare -A said
note() {
  [[ "${said[$1]:-}" == "$2" ]] && return
  said[$1]="$2"
  log "$1: $2"
}

# Expect $SCRATCH/gitlab-runner/f7t/<slot>/<project>; touch nothing on any other layout.
if [[ "${slots_root}" != */gitlab-runner/f7t ]]; then
  log "unexpected job dir layout (${CI_PROJECT_DIR}), skipping"
  exit 0
fi

# Whether CI job $1 still holds its slot. Any squeue state but COMPLETING counts;
# if squeue itself fails, assume it does.
job_active() {
  local state
  state="$(squeue -h -u "${me}" -n "ci-$1" -o %T 2>/dev/null)" || return 0
  state="${state%%$'\n'*}"
  [[ -n "${state}" && "${state}" != COMPLETING ]]
}

# Whether claim $2 (.swept.<sweeper>.<name>) in slot $1 was left by a sweeper
# that is gone.
claim_abandoned() {
  local rest="${2#.swept.}"
  local by="${rest%%.*}"
  if [[ "${by}" =~ ^[0-9]+$ ]]; then
    ! job_active "${by}"
  else
    [[ -z "$(find "$1/$2" -maxdepth 0 -cmin -"${MANUAL_CLAIM_MIN}" 2>/dev/null)" ]]
  fi
}

# Claim the given entries of slot $1 by renaming them, then delete the claims
# in the background.
claim_and_delete() {
  local dir="$1" name base target claimed=()
  shift
  for name in "$@"; do
    base="${name}"
    if [[ "${name}" == .swept.* ]]; then
      base="${name#.swept.}"
      base="${base#*.}"
    fi
    target=".swept.${sweeper}.${base}"
    [[ -e "${dir}/${target}" ]] && continue
    # rename(2) is atomic: if another sweeper got here first, this one fails.
    mv -T -- "${dir}/${name}" "${dir}/${target}" 2>/dev/null && claimed+=("${target}")
  done
  if (( ${#claimed[@]} == 0 )); then
    log "${dir}: claimed by another sweeper first"
    return
  fi
  log "${dir}: claimed ${claimed[*]}, deleting"
  (
    for name in "${claimed[@]}"; do
      # "\;" not "+": Bazel's mode-000 sandbox dir must be opened up before find
      # descends into it. See cscs-mi300.md: "Job directory cleanup".
      find "${dir}/${name}" -type d ! -perm -u=rwx -exec chmod u+rwx {} \; 2>/dev/null
      rm -rf "${dir:?}/${name:?}" 2>/dev/null || log "${dir}/${name}: not fully removed"
    done
    log "${dir}: done"
  ) &
}

swept=0
sweep_pass() {
  local dir p e owner newest_script todo held
  # Without squeue there is no telling which slots are in use.
  if ! squeue -h -u "${me}" > /dev/null; then
    log "squeue failed, skipping this pass"
    return
  fi
  for dir in "${slots_root}"/*/"${project}"; do
    [[ -d "${dir}" && "${dir}" != "${CI_PROJECT_DIR}" ]] || continue
    # Left to delete: build dirs, and claims whose sweeper is gone.
    todo=()
    held=0
    for p in "${BUILD_DIRS[@]}"; do
      [[ -e "${dir}/${p}" || -L "${dir}/${p}" ]] && todo+=("${p}")
    done
    for e in "${dir}"/.swept.*; do
      [[ -e "${e}" || -L "${e}" ]] || continue
      if claim_abandoned "${dir}" "$(basename "${e}")"; then
        todo+=("$(basename "${e}")")
      else
        held=1
      fi
    done
    if (( ${#todo[@]} == 0 )); then
      (( held )) && note "${dir}" "deletion in progress"
      continue
    fi

    newest_script="$(ls -t "${dir}"/script_* 2>/dev/null | head -n 1)"
    owner="$(grep -oE 'CI_JOB_ID=[0-9]+' "${newest_script:-/dev/null}" 2>/dev/null | head -n 1 | cut -d= -f2)"
    if [[ -z "${owner}" ]]; then
      note "${dir}" "owning job unknown, skipping"
      continue
    fi
    if job_active "${owner}"; then
      note "${dir}" "job ${owner} still active, skipping"
      continue
    fi
    if [[ -n "$(find "${dir}" -maxdepth 1 -mmin -"${STALE_MIN}" -print -quit 2>/dev/null)" ]]; then
      note "${dir}" "job ${owner} is gone, waiting until nothing changed for ${STALE_MIN} min"
      continue
    fi

    said[${dir}]=""
    swept=$((swept + 1))
    if [[ "${DRY_RUN}" == 1 ]]; then
      log "${dir}: job ${owner} is gone, would claim ${todo[*]}"
    else
      claim_and_delete "${dir}" "${todo[@]}"
    fi
  done
}

log "started for ${slots_root}/*/${project} (own slot excluded), every ${REPEAT_EVERY_S} s for ${REPEAT_MIN} min, stale after ${STALE_MIN} min"
end=$((SECONDS + REPEAT_MIN * 60))
while :; do
  sweep_pass
  (( SECONDS < end )) || break
  sleep "${REPEAT_EVERY_S}"
done
wait   # for deletions still running
log "finished, ${swept} slot(s) swept"
exit 0
