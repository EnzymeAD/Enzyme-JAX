# Copilot Coding Agent Instructions for Enzyme-JAX

## Repository Overview

Enzyme-JAX provides custom bindings for the Enzyme automatic differentiation (AD) tool integrated with JAX. It enables automatic importing and differentiation (JVP and VJP) of external C++ code into JAX. The project is language-agnostic via Enzyme, supporting Julia, Swift, Fortran, Rust, and Python.

**Project Type:** Python/C++ library with MLIR-based compiler passes  
**Build System:** Bazel 7.7.0  
**Languages:** Python 3.10-3.12, C++17  
**Key Dependencies:** JAX, XLA, LLVM/MLIR, Enzyme, StableHLO, Triton

## Repository Structure

```
/                       # Root: BUILD, WORKSPACE, workspace.bzl, package.bzl
├── src/enzyme_ad/jax/  # Main source code
│   ├── primitives.py   # Core Python API (cpp_call, enzyme_jax_ir, etc.)
│   ├── __init__.py     # Package exports
│   ├── Passes/         # MLIR optimization passes
│   ├── Implementations/# AD derivative implementations
│   ├── Dialect/        # Custom MLIR dialects (EnzymeXLA, Distributed, Tessera)
│   └── TransformOps/   # MLIR transform operations
├── test/               # Tests (Python and lit tests)
│   ├── lit_tests/      # MLIR FileCheck-based lit tests (.mlir, .pyt files)
│   ├── test.py         # Main Python tests
│   └── BUILD           # Test build configuration
├── builddeps/          # Python requirements
├── third_party/        # External dependency configurations (JAX, XLA, Enzyme, etc.)
└── patches/            # Patches for JAX and XLA
```

## Build Instructions

### Prerequisites
- Bazel 7.7.0 (specified in `.bazelversion`)
- clang++ (C++17 support required)
- Python 3.10, 3.11, or 3.12
- python-virtualenv, python3-dev

### Build Commands

**Build the wheel:**
```sh
bazel build :wheel
# Output: bazel-bin/enzyme_ad-VERSION-SYSTEM.whl
```

**Build enzymexlamlir-opt:**
```sh
bazel build //:enzymexlamlir-opt
```

**Update Python requirements (run before building if requirements change):**
```sh
bazel run //builddeps:requirements.update
```

**Generate compile_commands.json for LSP support:**
```sh
bazel run :refresh_compile_commands
```

### Environment Variables

Set `HERMETIC_PYTHON_VERSION` to specify Python version:
```sh
HERMETIC_PYTHON_VERSION=3.11 bazel build :wheel
```

For macOS builds, add: `--define using_clang=true`  
For Linux ARM64, add: `--linkopt=-fuse-ld=lld`

### Build Notes
- Builds can take 2-3+ hours on first run (downloads and compiles LLVM, XLA, etc.)
- Use `bazel-bin/` output directory for built artifacts
- Build artifacts are cached; subsequent builds are faster

## Testing

**Run all tests:**
```sh
bazel test //test/...
```

**Run lit tests only:**
```sh
bazel test //test/lit_tests/...
```

**Run specific test:**
```sh
bazel test //test:test  # Main Python tests
bazel test //test:testffi
bazel test //test:llama
```

**Test flags for verbose output:**
```sh
bazel test --test_output=errors --experimental_ui_max_stdouterr_bytes=-1 --test_verbose_timeout_warnings //test/...
```

**Important:** Do NOT run Python tests from the repository root. If testing manually after installing the wheel:
```sh
cd test && python test.py
```

## Code Formatting

### Python (Black)
- Style: Black formatter
- Check: `black --check .`
- Fix: `black .`
- CI workflow: `.github/workflows/python_format.yml`

### C++/Header Files (clang-format)
- Style: LLVM (see `.clang-format`)
- Source directory: `src/` (excludes `src/external/`)
- clang-format version: 16
- CI workflow: `.github/workflows/format.yml`

### Bazel Files (buildifier)
- Check: `buildifier -mode check -r .`
- Fix: `buildifier -r .`
- CI workflow: `.github/workflows/format-check-bazel.yml`

## CI/CD Workflows

Located in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `build.yml` | Push/PR to main | Full build and test on multiple platforms |
| `format.yml` | Push/PR (C++ files) | clang-format check |
| `python_format.yml` | Push/PR (Python files) | Black format check |
| `format-check-bazel.yml` | Push/PR (Bazel files) | Buildifier format check |

## Key Source Files

- **`src/enzyme_ad/jax/primitives.py`** - Main Python API: `cpp_call`, `ffi_call`, `hlo_call`, `enzyme_jax_ir`, `optimize_module`
- **`src/enzyme_ad/jax/__init__.py`** - Public API exports
- **`BUILD`** (root) - Main Bazel build file, defines `:wheel`, `:enzymexlamlir-opt`, `:enzyme_ad`
- **`WORKSPACE`** - Dependency loading and workspace setup
- **`workspace.bzl`** - Version pins for JAX, Enzyme, LLVM targets, XLA patches

## Adding/Modifying Code

### Python Changes
1. Edit files in `src/enzyme_ad/jax/`
2. Run `black .` to format
3. Test with `bazel test //test:test`

### C++ Changes
1. Edit files in `src/enzyme_ad/jax/` (.cpp, .h, .cc files)
2. Format with clang-format (LLVM style, version 16)
3. Build with `bazel build //:enzymexlamlir-opt` or `bazel build :wheel`

### Bazel File Changes
1. Edit BUILD, WORKSPACE, or .bzl files
2. Format with `buildifier -r .`
3. Verify with `bazel build :wheel`

### Adding MLIR Passes
- Passes are in `src/enzyme_ad/jax/Passes/`
- TableGen definitions in `.td` files
- Register in `src/enzyme_ad/jax/RegistryUtils.cpp`

## Dependencies

External dependencies are configured in `third_party/`:
- `enzyme/workspace.bzl` - Enzyme AD
- `jax/workspace.bzl` - Google JAX
- `xla/workspace.bzl` - OpenXLA
- `ml_toolchain/workspace.bzl` - ML toolchain

Version pins are in `workspace.bzl`:
- `JAX_COMMIT`, `ENZYME_COMMIT`, `ML_TOOLCHAIN_COMMIT`, `HEDRON_COMPILE_COMMANDS_COMMIT`

## Troubleshooting

- **Build timeout:** Initial builds can exceed 180 minutes. Increase timeout or use cached builds.
- **Python import errors:** Install wheel before running Python: `pip install bazel-bin/enzyme_ad-*.whl`
- **Format failures:** Run formatters locally before committing.
- **Test failures in lit_tests:** These are MLIR FileCheck tests; check `.mlir` and `.pyt` files.

## Trust These Instructions

These instructions are verified and accurate. Only perform additional searches if:
1. The instructions appear incomplete for your specific task
2. You encounter errors not documented here
3. File locations have changed from what's documented
