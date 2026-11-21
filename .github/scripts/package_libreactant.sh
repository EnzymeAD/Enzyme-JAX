#!/usr/bin/env bash
# Package libReactantExtra.so and its runtime/CUDA dependencies into a
# relocatable zstd-compressed tarball for sharing across CI jobs.
#
# Usage:
#   package_libreactant.sh <bazel_bin_dir> <output_tar_zst>

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <bazel_bin_dir> <output_tar_zst>" >&2
  exit 1
fi

BAZEL_BIN="$1"
OUTPUT_ARCHIVE="$2"

if [ -f "${BAZEL_BIN}/src/enzyme_ad/jax/ReactantExtra/libReactantExtra.so" ]; then
  REACTANT_BUILD_DIR="${BAZEL_BIN}/src/enzyme_ad/jax/ReactantExtra"
elif [ -f "${BAZEL_BIN}/external/enzyme_ad/src/enzyme_ad/jax/ReactantExtra/libReactantExtra.so" ]; then
  REACTANT_BUILD_DIR="${BAZEL_BIN}/external/enzyme_ad/src/enzyme_ad/jax/ReactantExtra"
else
  echo "ERROR: Could not find libReactantExtra.so under ${BAZEL_BIN}" >&2
  exit 1
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

install -m 755 "${REACTANT_BUILD_DIR}/libReactantExtra.so" "${STAGE_DIR}/libReactantExtra.so"

HAS_SOLIB=0
for solib_dir in "${BAZEL_BIN}"/_solib_local/*; do
  [ -d "${solib_dir}" ] || continue
  case "$(basename "${solib_dir}")" in
    *stub*) continue ;;
  esac
  for so_file in "${solib_dir}"/*.so*; do
    [ -e "${so_file}" ] || continue
    cp -fvL "${so_file}" "${STAGE_DIR}/"
    chmod 755 "${STAGE_DIR}/$(basename "${so_file}")"
    HAS_SOLIB=1
  done
done

# cuDNN dlopens libcudnn_engines_tensor_ir.so.9 at runtime when dynamically
# linked (e.g. CUDA 12), so Bazel does not place it in _solib_local.
if [ -f "${STAGE_DIR}/libcudnn_graph.so.9" ] && [ ! -f "${STAGE_DIR}/libcudnn_engines_tensor_ir.so.9" ]; then
  CUDNN_GRAPH_LINK="$(compgen -G "${BAZEL_BIN}/_solib_local/*/libcudnn_graph.so.9" | head -n 1 || true)"
  if [ -n "${CUDNN_GRAPH_LINK}" ]; then
    CUDNN_LIB_DIR="$(dirname "$(readlink -f "${CUDNN_GRAPH_LINK}")")"
    if [ -f "${CUDNN_LIB_DIR}/libcudnn_engines_tensor_ir.so.9" ]; then
      cp -fvL "${CUDNN_LIB_DIR}/libcudnn_engines_tensor_ir.so.9" "${STAGE_DIR}/"
      chmod 755 "${STAGE_DIR}/libcudnn_engines_tensor_ir.so.9"
    fi
  fi
fi

RUNFILES_DIR="${REACTANT_BUILD_DIR}/libReactantExtra.so.runfiles"
if [ -e "${RUNFILES_DIR}/cuda_nvcc/bin/ptxas" ]; then
  install -Dvm 755 "${RUNFILES_DIR}/cuda_nvcc/bin/ptxas" "${STAGE_DIR}/cuda/bin/ptxas"
  install -Dvm 755 "${RUNFILES_DIR}/cuda_nvcc/bin/fatbinary" "${STAGE_DIR}/cuda/bin/fatbinary"
  if [ -e "${RUNFILES_DIR}/cuda_nvvm/nvvm/libdevice/libdevice.10.bc" ]; then
    install -Dvm 644 "${RUNFILES_DIR}/cuda_nvvm/nvvm/libdevice/libdevice.10.bc" "${STAGE_DIR}/cuda/nvvm/libdevice/libdevice.10.bc"
  elif [ -e "${RUNFILES_DIR}/cuda_nvcc/nvvm/libdevice/libdevice.10.bc" ]; then
    install -Dvm 644 "${RUNFILES_DIR}/cuda_nvcc/nvvm/libdevice/libdevice.10.bc" "${STAGE_DIR}/cuda/nvvm/libdevice/libdevice.10.bc"
  else
    echo "ERROR: libdevice.10.bc not found under ${RUNFILES_DIR}" >&2
    exit 1
  fi
fi

if [ "${HAS_SOLIB}" -eq 1 ] || readelf -d "${STAGE_DIR}/libReactantExtra.so" | grep -q -E '\((RPATH|RUNPATH)\)'; then
  if ! command -v patchelf >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update && apt-get install -y --no-install-recommends patchelf
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y --no-install-recommends patchelf
    else
      echo "ERROR: patchelf is required to rewrite RPATH on libReactantExtra.so" >&2
      exit 1
    fi
  fi
  patchelf --set-rpath '$ORIGIN' "${STAGE_DIR}/libReactantExtra.so"
fi

# Verify that every non-system dynamic dependency is bundled alongside libReactantExtra.so.
provided="$(
  {
    find "${STAGE_DIR}" -maxdepth 1 -name '*.so*' -printf '%f\n'
    find "${STAGE_DIR}" -maxdepth 1 -name '*.so*' -type f -exec readelf -d {} + \
      | sed -n 's/.*(SONAME).*\[\(.*\)\].*/\1/p'
  } | sort -u
)"
unbundled=0
for lib in "${STAGE_DIR}"/*.so*; do
  [ -f "${lib}" ] || continue
  for need in $(readelf -d "${lib}" | sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p'); do
    case "${need}" in
      libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libresolv.so.*|libgcc_s.so.*|libstdc++.so.*|ld-linux*.so.*|libcuda.so.*)
        continue
        ;;
    esac
    if ! grep -qxF "${need}" <<< "${provided}"; then
      echo "ERROR: $(basename "${lib}") requires ${need}, which is not in bundle" >&2
      unbundled=1
    fi
  done
done
if [ "${unbundled}" -ne 0 ]; then
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ARCHIVE}")"
tar -I 'zstd -T0 -3' -cf "${OUTPUT_ARCHIVE}" -C "${STAGE_DIR}" .
ls -lh "${OUTPUT_ARCHIVE}"
