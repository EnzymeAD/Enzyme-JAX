#!/bin/bash
# One-time (idempotent) preparation of the local build tree: bazel shims, pinned
# checkouts of Reactant.jl and GB-25, and the Reactant build environment.
#
# Runs on the LOGIN NODE. Nothing here needs a GPU or the uenv; keeping it out of
# the batch job shrinks the 48-minute job's failure surface to the build itself.
#
#   source ci/local/env.sh && ci/local/setup.sh
set -euo pipefail
: "${BUILD_ROOT:?source ci/local/env.sh first}"

mkdir -p "${BUILD_ROOT}"/{bin,logs,src,.julia,.bazelisk,.bazel,.bazel-disk,.rocm}

# --- bazel/bazelisk shims: pin --output_user_root so a build never fills $HOME.
# Verbatim from ci/cscs-mi300.yml. See cscs-mi300.md: "Bazel output root".
if [[ ! -x "${LOCAL_BIN}/bazelisk-real" ]]; then
  curl -fsSL -o "${LOCAL_BIN}/bazelisk-real" \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
  chmod +x "${LOCAL_BIN}/bazelisk-real"
fi
for CMD in bazelisk bazel; do
  cat > "${LOCAL_BIN}/${CMD}" << 'WRAPPER'
#!/bin/bash
set -euo pipefail
if [[ -z "${BAZEL_OUTPUT_ROOT:-}" ]]; then
  echo "ERROR: BAZEL_OUTPUT_ROOT is not set. Refusing to run (would write to ~/.cache/bazel)." >&2
  exit 1
fi
if [[ "${BAZEL_OUTPUT_ROOT}" != "${CI_PROJECT_DIR}"* ]]; then
  echo "ERROR: BAZEL_OUTPUT_ROOT (${BAZEL_OUTPUT_ROOT}) is not under CI_PROJECT_DIR (${CI_PROJECT_DIR})." >&2
  exit 1
fi
exec "$(dirname "$0")/bazelisk-real" --output_user_root="${BAZEL_OUTPUT_ROOT}" "$@"
WRAPPER
  chmod +x "${LOCAL_BIN}/${CMD}"
  RESOLVED=$(command -v "${CMD}")
  [[ "${RESOLVED}" == "${LOCAL_BIN}/${CMD}" ]] || { echo "ERROR: ${CMD} resolves to ${RESOLVED}" >&2; exit 1; }
  echo "OK: ${CMD} -> ${RESOLVED}"
done

# --- pinned checkouts. --branch takes a ref, not a SHA, so fetch the SHA directly.
pin_checkout() {  # url dir sha
  local url="$1" dir="$2" sha="$3"
  if [[ -d "${dir}/.git" ]] && [[ "$(git -C "${dir}" rev-parse HEAD 2>/dev/null)" == "${sha}" ]]; then
    echo "OK: ${dir##*/} already at ${sha:0:12}"; return
  fi
  rm -rf "${dir}"; mkdir -p "${dir}"
  git -C "${dir}" init -q
  git -C "${dir}" remote add origin "${url}"
  git -C "${dir}" fetch -q --depth=1 origin "${sha}" \
    || { echo "  shallow SHA fetch refused, falling back to full clone"; git -C "${dir}" fetch -q origin; }
  git -C "${dir}" checkout -q "${sha}"
  echo "OK: ${dir##*/} at $(git -C "${dir}" rev-parse --short HEAD)"
}
pin_checkout https://github.com/EnzymeAD/Reactant.jl "${REACTANT_DIR}" "${REACTANT_COMMIT}"
pin_checkout https://github.com/PRONTOLab/GB-25    "${GB25_DIR}"     "${GB25_COMMIT}"

# --- point Reactant's Bazel WORKSPACE at our pinned Enzyme-JAX.
sed -i 's/ENZYMEXLA_COMMIT = ".*"/ENZYMEXLA_COMMIT = "'"${ENZYMEXLA_COMMIT}"'"/' \
  "${REACTANT_DIR}/deps/ReactantExtra/WORKSPACE"
grep -q "ENZYMEXLA_COMMIT = \"${ENZYMEXLA_COMMIT}\"" "${REACTANT_DIR}/deps/ReactantExtra/WORKSPACE" \
  || { echo "ERROR: ENZYMEXLA_COMMIT rewrite did not take" >&2; exit 1; }
echo "OK: ENZYMEXLA_COMMIT -> ${ENZYMEXLA_COMMIT:0:12}"

# --- Reactant build environment (no GPU / uenv needed)
cd "${REACTANT_DIR}/deps" && "${JULIA}" --project --color=yes -e 'using Pkg; Pkg.instantiate()'
echo "OK: setup complete — next: sbatch ci/local/build.sbatch"
