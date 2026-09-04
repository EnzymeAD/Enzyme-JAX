// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%comm : !comm.mpi.comm) {
    comm.mpi.barrier %comm : !comm.mpi.comm
    return
}
// SHLO: func.func @main(%[[COMM:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiBarrier(%[[COMM]]) {has_side_effect = true} : (tensor<i64>) -> ()

// JIT-LABEL: llvm.func @MPI_Barrier
// JIT-SAME:                        (!llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Barrier
// JIT-SAME:                                          (%arg0: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[COMM:.*]]: tensor<i64>) {
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Barrier (%[[COMM]]) : (tensor<i64>) -> ()
