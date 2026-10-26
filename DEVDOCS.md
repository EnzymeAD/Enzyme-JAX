# Enzyme-JAX Developer Notes

## Development Environment

The `.devcontainer` configuration is helpful but not recommended. While it provides a pre-configured development environment with all necessary dependencies, tools, and settings already installed, you may choose to set up your environment manually if preferred.

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
- **AffineCFG.cpp** - Perform optimizations and control-flow raisings using affine constructs.
- **AffineToStableHLORaising.cpp** - Raises affine loop nests to tensor operations using the stablehlo
  dialect.
- **ArithRaising.cpp** - Raises arith and math operations on tensors to the stablehlo / chlo dialects.
- **CanonicalizeFor.cpp** & **CanonicalizeLoops.cpp** - Raise scf.while to scf.for loops and other
  canonicalizations.
- **ConsumingInterpreterPass.cpp** - Apply transform ops patterns to the module.
- **ControlFlowToSCF.cpp** - Upstream ControlFlowToSCF checked in.
- **DelinearizeIndexing.cpp** - Transforms linear memory accesses to affine expressions.
- **DropUnsupportedAttributesPass.cpp** - Removes internal enzymexla attributes that are not
  supported by XLA.
- **EnzymeHLOOpt.cpp** - Core optimization patterns for StableHLO and EnzymeXLA operations
  This file contains (nearly) all the stablehlo tensor optimizations. Other optimization patterns
  are written in Tablegen and located in **StablehloOptPatterns.td**.
- **EnzymeHLOUnroll.cpp** - Unroll stablehlo.while ops in for loop forms.
- **KernelCastPass.cpp** - Changes the floating type from GPU kernels to another floating point.
  Used to support number types such as bfloat16 when the frontend cannot generate correct code
  for it.
- **LibDeviceFuncsRaisingPass.cpp** - Raise device side function calls and llvm intrinsics to higher
  level MLIR operations of the relevant dialects (math, arith, nvvm).
- **LLVMToAffineAccess.cpp** - Transform linear loads and stores from the llvm dialect to affine
  constructs.
- **LLVMToControlFlow.cpp** - From llvm to control flow.
- **SimplifyAffineExprs.cpp** - Try to simplify affine expressions using the Integer Set Library (ISL).
- **SortMemory.cpp** - Sort memory accesses if non-overlapping.
- **SROAWrappers.cpp** - Calls the LLVM SROA pass on LLVM functions contained in the given MLIR module.

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
6. Add the pass to the appropriate pass list in `src/enzyme_ad/jax/Integrations/c/EnzymeXLA.cpp` using the name from `TransformOps.td`

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
This recursively runs all targets in the `test/` directory.

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
bazel test //test/lit_tests/...
```
