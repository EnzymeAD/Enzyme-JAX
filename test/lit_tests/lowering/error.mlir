// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{backend=cpu},canonicalize)" | FileCheck %s

module @reactant_throw attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.mlir.global external constant @error_msg("my custom error msg") {addr_space = 0 : i32}
  func.func @error() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @error_msg : !llvm.ptr
    return %0 : !llvm.ptr
  }
  func.func @main() {
    enzymexla.jit_call @error () : () -> ()
    return
  }
}
