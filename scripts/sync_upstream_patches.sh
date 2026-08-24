#!/usr/bin/env bash
# Re-download the patch files that xla_deps.MODULE.bazel applies to Bazel
# Central Registry modules (grpc, protobuf, abseil-cpp, rules_python).
#
# bzlmod only honours overrides declared by the root module, so the patches
# that @xla//:MODULE.bazel and @jax//:MODULE.bazel would apply to their BCR
# dependencies have to be re-declared here, and the patch files themselves
# must live in this repository.  This script keeps `patches/upstream/` in
# sync with the JAX and XLA commits pinned in `MODULE.bazel`.
#
# Usage: scripts/sync_upstream_patches.sh [path/to/MODULE.bazel]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_FILE="${1:-${REPO_ROOT}/MODULE.bazel}"
OUT_DIR="${REPO_ROOT}/patches/upstream"

get_var() {
    grep -E "^${1} *= *\"[0-9a-f]+\"" "${MODULE_FILE}" | head -1 | cut -d= -f2 | tr -d '" '
}

JAX_COMMIT="$(get_var JAX_COMMIT)"
XLA_COMMIT="$(get_var XLA_COMMIT)"

if [[ -z "${JAX_COMMIT}" || -z "${XLA_COMMIT}" ]]; then
    echo "Could not read JAX_COMMIT/XLA_COMMIT from ${MODULE_FILE}" >&2
    exit 1
fi

# Patches applied by @xla//:MODULE.bazel.
XLA_PATCHES=(
    third_party/grpc/grpc.patch
    third_party/protobuf/protobuf_arena.patch
    third_party/protobuf/fix_message_lite_incomplete_type.patch
    third_party/protobuf/fix_python_dist_package.patch
    third_party/absl/btree.patch
    third_party/absl/build_dll.patch
    third_party/absl/endian.patch
    third_party/absl/raw_hash_set.patch
)

# Patches applied by @jax//:MODULE.bazel (none at the moment: JAX's
# rules_python local-wheel patch does not apply to BCR's rules_python and we
# don't build against local wheels).
JAX_PATCHES=()

mkdir -p "${OUT_DIR}"

fetch() {
    local repo="$1" commit="$2" path="$3"
    local url="https://raw.githubusercontent.com/${repo}/${commit}/${path}"
    echo "Fetching ${url}"
    curl -fsSL --retry 3 -o "${OUT_DIR}/$(basename "${path}")" "${url}"
}

for p in "${XLA_PATCHES[@]}"; do
    fetch openxla/xla "${XLA_COMMIT}" "${p}"
done
for p in "${JAX_PATCHES[@]-}"; do
    [[ -z "${p}" ]] && continue
    fetch jax-ml/jax "${JAX_COMMIT}" "${p}"
done

echo "Upstream patches synced to ${OUT_DIR}"
