# Enzyme-JAX Developer Notes

## Building the Project

### Quick Build
```bash
bazel build --repo_env=CC=clang-18 --color=yes -c dbg :enzymexlamlir-opt
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
  This file contains (nearly) all the stablehlo tensor optimizations.

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
6. Add the pass to the appropriate pass list in `src/enzyme_ad/jax/primitives.py`

### Adding a new lowering pass
1. Define the pass in `src/enzyme_ad/jax/Passes/Passes.td`, e.g. `LowerEnzymeXLALinalgPass`
2. Create a new `.cpp` file in `src/enzyme_ad/jax/Passes/`, e.g. `src/enzyme_ad/jax/Passes/LowerEnzymeXLALinalg.cpp`. In the new file...
   1. Inherit from `mlir::OpRewritePattern<OpType>` and implement the `matchAndRewrite()` method.
   2. Inherit from the generated `PassBase` class and implement `runOnOperation` to register your pass.
3. Write lit tests for your pass, e.g. `test/lit_tests/linalg/*.mlir`.

### Adding a New Dialect Operation

1. Define operation in `src/enzyme_ad/jax/Dialect/EnzymeXLAOps.td`
2. Specify arguments, results, and traits
3. Implement operation class if needed in `src/enzyme_ad/jax/Dialect/Ops.cpp`
4. TODO: write about derivative rules?

## Testing

Run all tests with:
```bash
bazel test //test/...
```
This runs all the test targets in `test/BUILD`.

Most of the Enzyme-JaX tests use [lit](https://llvm.org/docs/CommandGuide/lit.html) for testing.
These tests are stored in `test/lit_tests`.
A lit test contains one or more run directives at the top a file.
e.g. in `test/lit_tests/if.mlir`:
```mlir
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
```
This instructs `lit` to run the `enzyme-hlo-opt` pass on `test/lit_tests/if.mlir`.
The output is fed to `FileCheck` which compares it against the expected result that is provided in comments in the file that start with `// CHECK`.

Run all lit tests with
```bash
bazel test //test/lit_tests:all
```
