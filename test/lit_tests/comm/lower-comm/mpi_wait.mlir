// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%req : !comm.mpi.request) {
    comm.mpi.wait %req : !comm.mpi.request
    return
}
// SHLO: func.func @main(%[[REQ:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiWait(%[[REQ]]) {has_side_effect = true} : (tensor<i64>) -> ()

// JIT-LABEL: llvm.func @MPI_Wait
// JIT-SAME:                     (!llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Wait
// JIT-SAME:                                       (%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.mlir.zero : !llvm.ptr
// JIT-NEXT:   %1 = llvm.call @MPI_Wait(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[REQ:.*]]: tensor<i64>) {
// JIT-NEXT: stablehlo.jit_call @enzymexla_jitwrap_MPI_Wait(%[[REQ]]) : (tensor<i64>) -> ()
