licenses(["notice"])

# Export the requirements files for use in other rules
exports_files([
    # Input files for each backend (includes common + backend-specific deps)
    "requirements-cuda12.in",
    "requirements-cuda13.in",
    "requirements-tpu.in",
    "requirements-cpu.in",
    # Common dependencies shared across all backends
    "test-requirements-common.txt",
    # Bazel backend detection rule
    "device_detect.bzl",
    # Overrides for dependency resolution
    "overrides.txt",
])

# Note: Lock files (requirements_lock_*.txt) are generated on CI
# and added to .gitignore. They are referenced dynamically in WORKSPACE.
