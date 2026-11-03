#ifndef ENZYMEXLA_PASSES_JULIA_PASSES_TD
#define ENZYMEXLA_PASSES_JULIA_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerStableHLOToJuliaPass : Pass<"lower-stablehlo-to-julia", "mlir::ModuleOp"> {
  let summary = "Lower StableHLO, CHLO, and Func operations to the Julia dialect";
  let dependentDialects = [
    "mlir::enzymexla::julia::JuliaDialect",
    "mlir::func::FuncDialect",
  ];
  let options = [
    Option<"reverse_dims", "reverse-dims", "bool", /*default=*/"true",
           "Reverse tensor dimensions so row-major StableHLO minor-most "
           "dimensions become column-major Julia dimension 0 (Reactant layout)">,
    Option<"prefer_blas", "prefer-blas", "bool", /*default=*/"true",
           "Lower compatible contractions, Cholesky, and triangular solves to "
           "Julia LinearAlgebra operations">,
  ];
}

def JuliaFuseOpsPass : Pass<"julia-fuse-ops", "mlir::ModuleOp"> {
  let summary = "Fuse elementwise, broadcast, transpose, slice, reverse, and "
                "matmul-epilogue operations in the Julia dialect";
  let dependentDialects = [
    "mlir::enzymexla::julia::JuliaDialect",
  ];
  let options = [
    Option<"fuse_horizontal", "fuse-horizontal", "bool", /*default=*/"true",
           "Horizontally merge compatible sibling `julia.kernel` loops">,
    Option<"fuse_matmul_epilogue", "fuse-matmul-epilogue", "bool", /*default=*/"true",
           "Fuse alpha/beta/accumulator arithmetic into 5-arg `julia.matmul`">,
  ];
}

def JuliaBufferizePass : Pass<"julia-bufferize", "mlir::ModuleOp"> {
  let summary = "Destination-passing bufferization, dead-operand in-place "
                "reuse, concat slice forwarding, while ping-pong hoisting, "
                "and linear-scan buffer pool coloring for the Julia dialect";
  let dependentDialects = [
    "mlir::enzymexla::julia::JuliaDialect",
  ];
  let options = [
    Option<"reuse_buffers", "reuse-buffers", "bool", /*default=*/"true",
           "Reuse dead owned buffers in-place and pool non-overlapping "
           "temporary allocations">,
    Option<"hoist_loop_allocs", "hoist-loop-allocs", "bool", /*default=*/"true",
           "Hoist loop-carried ping-pong buffers and scratch allocations out "
           "of `julia.while` loops">,
    Option<"forward_concat_slices", "forward-concat-slices", "bool", /*default=*/"true",
           "Write single-use `julia.concat` inputs directly into `@view` "
           "slices of the destination array">,
  ];
}

def EmitJuliaPass : Pass<"emit-julia", "mlir::ModuleOp"> {
  let summary = "Emit native Julia source code for the module";
  let dependentDialects = [
    "mlir::enzymexla::julia::JuliaDialect",
  ];
  let options = [
    Option<"output", "output", "std::string", /*default=*/"\"\"",
           "Optional output file path (writes to stdout if empty)">,
  ];
}

#endif // ENZYMEXLA_PASSES_JULIA_PASSES_TD
