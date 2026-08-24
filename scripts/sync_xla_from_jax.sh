#!/usr/bin/env bash
# Align the XLA and rules_ml_toolchain pins in MODULE.bazel with the
# ones used by the given JAX commit, then refresh the vendored upstream
# patches.  Run after bumping JAX_COMMIT.
#
# Usage: scripts/sync_xla_from_jax.sh <jax_commit> [path/to/MODULE.bazel]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAX_COMMIT="${1:?usage: $0 <jax_commit> [MODULE.bazel]}"
MODULE_FILE="${2:-${REPO_ROOT}/MODULE.bazel}"
JAX_RAW="https://raw.githubusercontent.com/jax-ml/jax/${JAX_COMMIT}"

XLA_COMMIT="$(curl -fsSL --retry 3 "${JAX_RAW}/third_party/xla/revision.bzl" | grep -oE 'XLA_COMMIT *= *"[0-9a-f]+"' | grep -oE '[0-9a-f]{40}')"
RULES_ML_TOOLCHAIN_COMMIT="$(curl -fsSL --retry 3 "${JAX_RAW}/MODULE.bazel" | grep -oE 'rules_ml_toolchain-[0-9a-f]{40}' | head -1 | cut -d- -f2)"

if [[ -z "${XLA_COMMIT}" || -z "${RULES_ML_TOOLCHAIN_COMMIT}" ]]; then
    echo "Could not determine XLA/rules_ml_toolchain commits from JAX ${JAX_COMMIT}" >&2
    exit 1
fi

echo "JAX ${JAX_COMMIT} uses XLA ${XLA_COMMIT} and rules_ml_toolchain ${RULES_ML_TOOLCHAIN_COMMIT}"
sed -i.bak -e "s/^JAX_COMMIT = \".*\"/JAX_COMMIT = \"${JAX_COMMIT}\"/" \
    -e "s/^XLA_COMMIT = \".*\"/XLA_COMMIT = \"${XLA_COMMIT}\"/" \
    -e "s/^RULES_ML_TOOLCHAIN_COMMIT = \".*\"/RULES_ML_TOOLCHAIN_COMMIT = \"${RULES_ML_TOOLCHAIN_COMMIT}\"/" \
    "${MODULE_FILE}"
rm -f "${MODULE_FILE}.bak"

"${REPO_ROOT}/scripts/sync_upstream_patches.sh" "${MODULE_FILE}"
