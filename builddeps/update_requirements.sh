#!/usr/bin/env bash
# Script to update Python requirements lock files using uv
# Usage: ./builddeps/update_requirements.sh [variant]
#
# Generates lock files for backends:
#   - requirements_lock_3_XX_cuda12.txt  (CUDA 12 support)
#   - requirements_lock_3_XX_cuda13.txt  (CUDA 13 support)
#   - requirements_lock_3_XX_tpu.txt     (TPU support)
#   - requirements_lock_3_XX_cpu.txt     (CPU-only)
#
# If a variant is specified (cuda12, cuda13, tpu, cpu), only that variant is generated.
# Use 'cuda' to auto-detect CUDA version.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or: pip install uv"
    exit 1
fi

echo "Using uv version: $(uv --version)"

# Function to detect CUDA version
detect_cuda_version() {
    # Try nvcc --version
    if command -v nvcc &> /dev/null; then
        local nvcc_output
        nvcc_output=$(nvcc --version 2>/dev/null | grep 'release' || true)
        if [[ "$nvcc_output" == *"release 13"* ]]; then
            echo "13"
            return
        elif [[ "$nvcc_output" == *"release 12"* ]]; then
            echo "12"
            return
        fi
    fi

    # Try /usr/local/cuda/version.txt
    if [[ -f /usr/local/cuda/version.txt ]]; then
        local version_content
        version_content=$(cat /usr/local/cuda/version.txt)
        if [[ "$version_content" == *"13."* ]]; then
            echo "13"
            return
        elif [[ "$version_content" == *"12."* ]]; then
            echo "12"
            return
        fi
    fi

    # Default to CUDA 12
    echo "12"
}

# Common arguments for uv pip compile
COMMON_ARGS=(
    "--generate-hashes"
    "--emit-index-url"
    "--no-strip-extras"
)

# Python versions to support
PYTHON_VERSIONS=("3.11" "3.12" "3.13")

# All available variants
ALL_VARIANTS=("cuda12" "cuda13" "tpu" "cpu")

# Function to compile requirements for a specific Python version and variant
compile_requirements() {
    local python_version="$1"
    local variant="$2"
    local input_file="requirements-${variant}.in"
    local output_file="requirements_lock_${python_version//./_}_${variant}.txt"

    if [[ ! -f "$input_file" ]]; then
        echo "Error: Input file $input_file not found"
        return 1
    fi

    echo "Compiling $variant requirements for Python $python_version -> $output_file"

    # TPU requires special handling for find-links
    local extra_args=()
    if [[ "$variant" == "tpu" ]]; then
        extra_args+=("--find-links" "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
    fi

    uv pip compile \
        "$input_file" \
        --python-version "$python_version" \
        --output-file "$output_file" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}" \
        --override overrides.txt \
        --custom-compile-command="./builddeps/update_requirements.sh"

    echo "Successfully generated $output_file"
}

# Determine which variants to generate
if [[ $# -gt 0 ]]; then
    input_variant="$1"

    # Handle 'cuda' as auto-detect
    if [[ "$input_variant" == "cuda" ]]; then
        cuda_ver=$(detect_cuda_version)
        echo "Auto-detected CUDA version: $cuda_ver"
        VARIANTS=("cuda${cuda_ver}")
    else
        VARIANTS=("$input_variant")
        # Validate variant
        valid=false
        for v in "${ALL_VARIANTS[@]}"; do
            if [[ "$input_variant" == "$v" ]]; then
                valid=true
                break
            fi
        done
        if [[ "$valid" == "false" ]]; then
            echo "Error: Invalid variant '$input_variant'. Must be one of: ${ALL_VARIANTS[*]} or 'cuda' (auto-detect)"
            exit 1
        fi
    fi
else
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

echo "=== Updating Python requirements lock files using uv ==="
echo "Variants: ${VARIANTS[*]}"
echo ""

# Flush out existing lock files for selected variants before regenerating
echo "Removing existing lock files for selected variants..."
for variant in "${VARIANTS[@]}"; do
    rm -f requirements_lock_3_*_${variant}.txt
done
echo ""

# Generate lock files
for version in "${PYTHON_VERSIONS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        compile_requirements "$version" "$variant"
    done
    echo ""
done

echo "=== All requirements lock files updated successfully! ==="
echo ""
echo "Generated files:"
for version in "${PYTHON_VERSIONS[@]}"; do
    version_underscore="${version//./_}"
    for variant in "${VARIANTS[@]}"; do
        echo "  - requirements_lock_${version_underscore}_${variant}.txt"
    done
done
echo ""
echo "Usage in Bazel:"
echo "  bazel build --config=cuda12 ...  # Use CUDA 12 lock files"
echo "  bazel build --config=cuda13 ...  # Use CUDA 13 lock files"
echo "  bazel build --config=tpu ...     # Use TPU lock files"
echo "  bazel build --config=cpu ...     # Use CPU-only lock files"
