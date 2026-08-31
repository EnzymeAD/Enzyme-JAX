#!/bin/bash
# Julia with LD_LIBRARY_PATH pointed at the curated ROCm libs, and nothing else.
# Appending the uenv's own lib dir here re-exposes the old libcurl/libstdc++ that
# shadow Julia's bundled copies -> "curl_easy_setopt: 48".
# See cscs-mi300.md: "Curated runtime libs", "run_julia.sh".
set -euo pipefail
: "${BUILD_ROOT:?source ci/local/env.sh first}"
export LD_LIBRARY_PATH="${BUILD_ROOT}/.rocm/reactant_libs"
exec "${JULIA}" "$@"
