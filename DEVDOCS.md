# Enzyme-JAX Developer Documentation

## Overview

Enzyme-JAX is a C++ project that integrates the Enzyme automatic differentiation tool with JAX, enabling automatic differentiation of external C++ code within JAX. The project uses LLVM's MLIR framework for intermediate representation and transformation of code.

## Building the Project

### Quick Build
```bash
bazel build --repo_env=CC=clang-18 --color=yes --copt=-fbracket-depth=1024 --host_copt=-fbracket-depth=1024 -c dbg :enzymexlamlir-opt
```

### Build Artifacts
- **Main tool**: `enzymexlamlir-opt` (bazel target: `:enzymexlamlir-opt`)
  - This is the MLIR optimization tool driver for Enzyme-XLA
  - Analogous to `mlir-opt`, drives compiler passes and transformations
  - Located in: `src/enzyme_ad/jax/enzymexlamlir-opt.cpp`

- **Python wheel**: `bazel build :wheel`

### Generate LSP Support
```bash
bazel run :refresh_compile_commands
```

## Project Structure

### Core Components

#### 1. **Dialects** (`src/enzyme_ad/jax/Dialect/`)
MLIR dialects define custom operations and types for Enzyme-JAX.

- **EnzymeXLAOps.td** - Dialect operation definitions
  - GPU operations: `kernel_call`, `memcpy`, `gpu_wrapper`, `gpu_block`, `gpu_thread`
  - JIT/XLA operations: `jit_call`, `xla_wrapper`
  - Linear algebra (BLAS/LAPACK): `symm`, `syrk`, `trmm`, `lu`, `getrf`, `gesvd`, etc.
  - Special functions: Bessel functions, GELU, ReLU
  - Utility operations: `memref2pointer`, `pointer2memref`, `subindex`

- **EnzymeXLAAttrs.td** - Custom attribute definitions (LAPACK enums, etc.)

#### 2. **Passes** (`src/enzyme_ad/jax/Passes/`)
MLIR passes implement transformations and optimizations.

- Tablegen definitions in `src/enzyme_ad/jax/Passes/Passes.td`
- **EnzymeHLOOpt.cpp** - Core optimization patterns for StableHLO and EnzymeXLA operations

#### 3. **Transform Operations** (`src/enzyme_ad/jax/TransformOps/`)
In order to have more granular control over which pattern is applied, patterns are also registered as transform operations.
For example:
```
def AndPadPad : EnzymeHLOPatternOp<
    "and_pad_pad"> {
  let patterns = ["AndPadPad"];
}
```
Exposes the `AndPadPad` pattern (defined in `EnzymeHLOOpt.cpp`) to `enzymexlamlir-opt`, so it can be used as:
```
enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=and_pad_pad" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect input.mlir
```

## Common Development Tasks

### Adding a New Optimization Pattern

1. Define the pattern class in `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`
2. Inherit from `mlir::OpRewritePattern<OpType>`
3. Implement `matchAndRewrite()` method
4. Register in `EnzymeHLOOptPass::runOnOperation()`
5. Register as Transform operation in `TransformOps.td`

### Adding a New Dialect Operation

1. Define operation in `src/enzyme_ad/jax/Dialect/EnzymeXLAOps.td`
2. Specify arguments, results, and traits
3. Implement operation class if needed in `src/enzyme_ad/jax/Dialect/Ops.cpp`
4. TODO: write about derivative rules?

## Testing

Run tests with:
```bash
bazel test //test/...
```
