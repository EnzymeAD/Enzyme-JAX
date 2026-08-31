#!/bin/bash
# Build libReactantExtra.so for MI300 (gfx942) against a CSCS uenv ROCm toolchain.
#
# Assembles the writable ROCm overlay at ${BUILD_ROOT}/.rocm, then runs Bazel via
# Reactant's build_local.jl. Must run INSIDE the uenv (srun --uenv / uenv run),
# where /user-environment is mounted.
#
# The overlay logic below was extracted verbatim from the BUILDSCRIPT heredoc that
# ci/cscs-mi300.yml used to generate inline; ci/cscs-mi300.md documents every
# workaround in it. Keeping it as a real file removes one heredoc level and lets
# CI and a local build run the same bytes.
#
# Required env:
#   BUILD_ROOT    root for .rocm overlay, curated libs, wrappers  (CI: $CI_PROJECT_DIR)
#   REACTANT_DIR  Reactant.jl checkout (build_local.jl lives in deps/)
#   JULIA         julia binary
# Optional env:
#   BAZEL_DISK_CACHE   dir for --disk_cache; makes incremental rebuilds cheap
#   ENZYMEXLA_LOCAL    path to a local Enzyme-JAX tree to build instead of the
#                      pinned GitHub tarball (see "local source mode" below)
set -euo pipefail

: "${BUILD_ROOT:?BUILD_ROOT must be set (root for the .rocm overlay)}"
: "${REACTANT_DIR:?REACTANT_DIR must be set (Reactant.jl checkout)}"
: "${JULIA:?JULIA must be set (path to the julia binary)}"
[[ -d /user-environment ]] || { echo "ERROR: /user-environment not mounted — run inside srun --uenv / uenv run" >&2; exit 1; }

# Seed include/ from the view, minus toolchain headers. See md: "LLVM headers".
mkdir -p "${BUILD_ROOT}/.rocm/include"
for f in /user-environment/env/default/include/*; do
  [[ -e "$f" ]] || continue
  name="$(basename "$f")"
  case "${name}" in
    llvm|llvm-c|mlir|mlir-c|clang|clang-c|lld|lldb|polly) continue ;;
  esac
  ln -sfn "$f" "${BUILD_ROOT}/.rocm/include/${name}"
done
# MIOpen and rocm-smi-lib are separate spack packages, not in the view.
# See md: "rocm-smi header".
for pkg in /user-environment/linux-zen3/miopen-hip-*/; do
  [[ -f "${pkg}include/miopen/version.h" ]] || continue
  ln -sfn "${pkg}include/miopen" "${BUILD_ROOT}/.rocm/include/miopen"
  export CPATH="${pkg}include${CPATH:+:${CPATH}}"
  break
done
if [[ ! -e "${BUILD_ROOT}/.rocm/include/rocm_smi/rocm_smi.h" ]]; then
  for pkg in /user-environment/linux-zen3/rocm-smi-lib-*/              /user-environment/linux-zen3/rocm-smi-*/              /user-environment/linux-zen3/*/; do
    [[ -f "${pkg}include/rocm_smi/rocm_smi.h" ]] || continue
    ln -sfn "${pkg}include/rocm_smi" "${BUILD_ROOT}/.rocm/include/rocm_smi"
    export CPATH="${pkg}include${CPATH:+:${CPATH}}"
    echo "rocm_smi -> ${pkg}include/rocm_smi"
    break
  done
fi
# amdgcn and bin are deliberately NOT whole-dir symlinks.
# See md: "amdgcn must stay a real dir", "clang wrapper + HIP_CLANG_PATH".
for d in lib64 share; do
  [[ -e "/user-environment/env/default/${d}" ]] &&     ln -sfn "/user-environment/env/default/${d}" "${BUILD_ROOT}/.rocm/${d}"
done
mkdir -p "${BUILD_ROOT}/.rocm/bin"
for f in /user-environment/env/default/bin/*; do
  [[ -e "$f" ]] || continue
  name="$(basename "$f")"
  [[ -e "${BUILD_ROOT}/.rocm/bin/${name}" ]] ||     ln -sfn "$f" "${BUILD_ROOT}/.rocm/bin/${name}"
done
# Device bitcode is needed at two paths. See md: "Device bitcode (two locations)".
DEVICE_LIB_DIR=""
if [[ -f "${BUILD_ROOT}/.rocm/amdgcn/bitcode/ocml.bc" ]]; then
  DEVICE_LIB_DIR="$(dirname "$(readlink -f "${BUILD_ROOT}/.rocm/amdgcn/bitcode/ocml.bc")")"
else
  for candidate in     /user-environment/env/default/amdgcn/bitcode     /user-environment/env/default/lib/llvm/lib/clang/*/amdgcn/bitcode     /user-environment/env/default/lib/clang/*/amdgcn/bitcode     /user-environment/linux-zen3/rocm-device-libs-*/amdgcn/bitcode     /user-environment/linux-zen3/rocm-device-libs-*/lib/amdgcn/bitcode     /user-environment/linux-zen3/llvm-amdgpu-*/lib/clang/*/amdgcn/bitcode; do
    [[ -f "${candidate}/ocml.bc" ]] || continue
    DEVICE_LIB_DIR="${candidate}"
    break
  done
fi
if [[ -n "${DEVICE_LIB_DIR}" ]]; then
  for parent in "${BUILD_ROOT}/.rocm/amdgcn" "${BUILD_ROOT}/.rocm/lib/llvm/amdgcn"; do
    [[ -L "${parent}" ]] && rm -f "${parent}"
    mkdir -p "${parent}"
  done
  rm -f "${BUILD_ROOT}/.rocm/amdgcn/bitcode" "${BUILD_ROOT}/.rocm/lib/llvm/amdgcn/bitcode"
  ln -s "${DEVICE_LIB_DIR}" "${BUILD_ROOT}/.rocm/amdgcn/bitcode"
  ln -s "${DEVICE_LIB_DIR}" "${BUILD_ROOT}/.rocm/lib/llvm/amdgcn/bitcode"
  echo "amdgcn/bitcode -> ${DEVICE_LIB_DIR}"
  echo "lib/llvm/amdgcn/bitcode -> ${DEVICE_LIB_DIR}"
else
  echo "WARNING: ROCm device libraries not found; HIP compilation may fail"
fi
# lib/ is per-file symlinks, view first then spack. See md: "Overlay lib/ layout".
mkdir -p "${BUILD_ROOT}/.rocm/lib"
for subdir in lib lib64; do
  for f in /user-environment/env/default/${subdir}/*.so*; do
    [[ -e "$f" ]] || continue
    name="$(basename "$f")"
    [[ -e "${BUILD_ROOT}/.rocm/lib/${name}" ]] ||       ln -sfn "$f" "${BUILD_ROOT}/.rocm/lib/${name}"
  done
done
for pkg in /user-environment/linux-zen3/*/; do
  for subdir in lib lib64; do
    [[ -d "${pkg}${subdir}" ]] || continue
    for item in "${pkg}${subdir}/"*.so*; do
      [[ -e "${item}" ]] || continue
      name="$(basename "${item}")"
      [[ -e "${BUILD_ROOT}/.rocm/lib/${name}" ]] ||         ln -sfn "${item}" "${BUILD_ROOT}/.rocm/lib/${name}"
    done
  done
done
# Overlay-local clang resource dir. See md: "clang wrapper + HIP_CLANG_PATH".
REAL_CLANG_BIN="$(which clang 2>/dev/null || true)"
if [[ -z "${REAL_CLANG_BIN}" || ! -x "${REAL_CLANG_BIN}" ]]; then
  echo "ERROR: clang not found in PATH; llvm/bin/clang wrapper not created" >&2
  exit 1
fi
REAL_CLANGXX_BIN="$(dirname "${REAL_CLANG_BIN}")/clang++"
[[ -x "${REAL_CLANGXX_BIN}" ]] || REAL_CLANGXX_BIN="${REAL_CLANG_BIN}"
REAL_RESOURCE_DIR="$("${REAL_CLANG_BIN}" -print-resource-dir)"
CLANG_RESOURCE_VERSION="$(basename "${REAL_RESOURCE_DIR}")"
OVERLAY_RESOURCE_DIR="${BUILD_ROOT}/.rocm/llvm/lib/clang/${CLANG_RESOURCE_VERSION}"
mkdir -p "${BUILD_ROOT}/.rocm/llvm/bin" "$(dirname "${OVERLAY_RESOURCE_DIR}")"
rm -rf "${OVERLAY_RESOURCE_DIR}"
cp -a "${REAL_RESOURCE_DIR}" "${OVERLAY_RESOURCE_DIR}"
# XLA's crosstool patch realpath's a hardcoded lib/llvm/lib/clang/22/include and
# dies if it is absent; alias it here. See md: "hipcc resource dir".
mkdir -p "${BUILD_ROOT}/.rocm/lib/llvm/lib/clang"
for v in "${CLANG_RESOURCE_VERSION}" 22; do
  ln -sfn "${OVERLAY_RESOURCE_DIR}" "${BUILD_ROOT}/.rocm/lib/llvm/lib/clang/${v}"
done
# One wrapper per driver, also linked into bin/ (HIP calls bin/clang++ directly,
# bypassing PATH). The wrapper forces -resource-dir and maps gold -> lld.
for drv in clang clang++; do
  if [[ "${drv}" == "clang" ]]; then real="${REAL_CLANG_BIN}"; else real="${REAL_CLANGXX_BIN}"; fi
  cat > "${BUILD_ROOT}/.rocm/llvm/bin/${drv}" << WRAPPER
#!/bin/bash
if [[ "\$#" -eq 1 && "\$1" == "-print-resource-dir" ]]; then
  printf '%s\n' "${OVERLAY_RESOURCE_DIR}"
  exit 0
fi
args=()
for arg in "\$@"; do
  if [[ "\${arg}" == "-fuse-ld=gold" ]]; then
    args+=("-fuse-ld=lld")
  else
    args+=("\${arg}")
  fi
done
exec "${real}" -resource-dir "${OVERLAY_RESOURCE_DIR}" "\${args[@]}"
WRAPPER
  chmod +x "${BUILD_ROOT}/.rocm/llvm/bin/${drv}"
  ln -sfn "${BUILD_ROOT}/.rocm/llvm/bin/${drv}" "${BUILD_ROOT}/.rocm/bin/${drv}"
done
CLANG_BIN="${BUILD_ROOT}/.rocm/llvm/bin/clang"
CLANGXX_BIN="${BUILD_ROOT}/.rocm/llvm/bin/clang++"
export PATH="${BUILD_ROOT}/.rocm/llvm/bin:${PATH}"
export HIP_CLANG_PATH="${BUILD_ROOT}/.rocm/llvm/bin"
# Overlay check (compact, always-on) — headers, device bitcode, hipcc resource
# dir, then the fatal shadow guard. See md: "Diagnostics".
echo "=== ROCm overlay check ==="
for f in rocm-core/rocm_version.h hip/hip_version.h miopen/version.h          rocblas/internal/rocblas-version.h rocrand/rocrand_version.h          rocfft/rocfft-version.h hipfft/hipfft-version.h          roctracer/roctracer.h hipsparse/hipsparse-version.h          hipsolver/internal/hipsolver-version.h rocsolver/rocsolver-version.h          rocm_smi/rocm_smi.h; do
  [[ -e "${BUILD_ROOT}/.rocm/include/${f}" ]] && echo "  ok   ${f}" || echo "  MISS ${f}"
done
for b in amdgcn/bitcode/ocml.bc lib/llvm/amdgcn/bitcode/ocml.bc          lib/llvm/lib/clang/22/include/stddef.h; do
  [[ -e "${BUILD_ROOT}/.rocm/${b}" ]] && echo "  ok   ${b}" || echo "  MISS ${b}"
done
for d in llvm mlir clang; do
  [[ -e "${BUILD_ROOT}/.rocm/include/${d}" ]] &&     { echo "FATAL: overlay include/${d} shadows XLA's LLVM/MLIR" >&2; exit 1; }
done
# Curate reactant_libs/ for run_julia.sh. readlink -f: JULIA may be a symlink,
# and the dirname chain must walk the real prefix. See md: "Curated runtime libs".
JULIA_LIBDIR="$(dirname "$(dirname "$(readlink -f "${JULIA}")")")/lib/julia"
CURATED="${BUILD_ROOT}/.rocm/reactant_libs"
rm -rf "${CURATED}"; mkdir -p "${CURATED}"
shopt -s nullglob
for f in "${BUILD_ROOT}/.rocm/lib/"*.so*; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  stem="${base%%.so*}"
  jmatch=("${JULIA_LIBDIR}/${stem}".so*)
  (( ${#jmatch[@]} )) && continue
  ln -sfn "$f" "${CURATED}/${base}"
done
shopt -u nullglob
echo "  curated $(ls -1 "${CURATED}" 2>/dev/null | wc -l) runtime libs"
echo "========================="
# PATH already leads with the wrapper dir (exported above).
# See md: "Bazel extraopts" for why each flag is here.

# Local source mode: build a local Enzyme-JAX tree instead of the pinned tarball.
# Bazel's --override_repository bypasses the http_archive's patch_cmds, so replay
# them on a throwaway copy (never on the user's working tree). Today those patch_cmds
# reduce to rewriting //:patches -> @enzyme_ad//:patches; the WORKSPACE's other,
# heavily-escaped sed matches nothing in workspace.bzl and is a no-op.
OVERRIDE_OPT=()
if [[ -n "${ENZYMEXLA_LOCAL:-}" ]]; then
  [[ -f "${ENZYMEXLA_LOCAL}/workspace.bzl" ]] || { echo "ERROR: ENZYMEXLA_LOCAL=${ENZYMEXLA_LOCAL} is not an Enzyme-JAX tree" >&2; exit 1; }
  OVERRIDE_SRC="${BUILD_ROOT}/src/enzyme_ad"
  rm -rf "${OVERRIDE_SRC}"; mkdir -p "${OVERRIDE_SRC}"
  tar -C "${ENZYMEXLA_LOCAL}" --exclude=.git -cf - . | tar -C "${OVERRIDE_SRC}" -xf -
  sed -i 's,//:patches,@enzyme_ad//:patches,g' "${OVERRIDE_SRC}"/third_party/*/workspace.bzl
  echo "enzyme_ad override -> ${OVERRIDE_SRC} (patch_cmds replayed)"
  OVERRIDE_OPT=(--extraopt="--override_repository=enzyme_ad=${OVERRIDE_SRC}")
fi

DISK_CACHE_OPT=()
[[ -n "${BAZEL_DISK_CACHE:-}" ]] && DISK_CACHE_OPT=(--extraopt="--disk_cache=${BAZEL_DISK_CACHE}")

# PATH already leads with the wrapper dir (exported above).
# See cscs-mi300.md: "Bazel extraopts" for why each flag is here.
cd "${REACTANT_DIR}/deps"
"${JULIA}" --project --color=yes -O0 build_local.jl \
  --cc=clang --gcc_host_compiler_path= \
  --extraopt="--repo_env=ROCM_PATH=${BUILD_ROOT}/.rocm" \
  --extraopt="--repo_env=TF_ROCM_AMDGPU_TARGETS=gfx942" \
  --extraopt="--action_env=PATH=${PATH}" \
  --extraopt="--action_env=HIP_CLANG_PATH=${BUILD_ROOT}/.rocm/llvm/bin" \
  --extraopt="--action_env=CLANGXX_COMPILER_PATH=${CLANGXX_BIN}" \
  --extraopt="--linkopt=-fuse-ld=lld" \
  --extraopt="--host_linkopt=-fuse-ld=lld" \
  --extraopt="--action_env=CLANG_COMPILER_PATH=${CLANG_BIN}" \
  "${DISK_CACHE_OPT[@]}" "${OVERRIDE_OPT[@]}"
