// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

module @nosideeffect {
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Barrier(!llvm.ptr) -> i32
  func.func @enzymexla_wrapper_MPI_Barrier() {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
    return
  }
  func.func @main() {
    // CHECK-NOT: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    return
  }
}

module @sideeffect {
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Barrier(!llvm.ptr) -> i32
  func.func @enzymexla_wrapper_MPI_Barrier() {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
    return
  }
  func.func @main() {
    // CHECK: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () {has_side_effect = true} : () -> ()
    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () {has_side_effect = true} : () -> ()
    return
  }
}
