// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=true},canonicalize)" %s | FileCheck %s --check-prefix=ASSUME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=false},canonicalize)" %s | FileCheck %s --check-prefix=NOASSUME

module {
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Barrier(!llvm.ptr) -> i32
  func.func @enzymexla_wrapper_MPI_Barrier() {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
    return
  }
  func.func @main() {
    // ASSUME-NOT: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    // NOASSUME: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    return
  }
}
