// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{backend=cpu},canonicalize,enzyme-hlo-opt)" | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{backend=cuda},canonicalize,enzyme-hlo-opt)" | FileCheck %s --check-prefix=CUDA

module @reactant_throw attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.mlir.global external constant @error_msg("my custom error msg") {addr_space = 0 : i32}
  func.func @error() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @error_msg : !llvm.ptr
    return %0 : !llvm.ptr
  }
  func.func @main() {
    // CPU: stablehlo.custom_call @enzymexla_compile_cpu_with_error()
    // CUDA: stablehlo.custom_call @enzymexla_compile_gpu_with_error()
    enzymexla.jit_call @error () : () -> ()
    return
  }
}
