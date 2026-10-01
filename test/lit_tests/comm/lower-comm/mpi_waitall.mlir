// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request) {
    comm.mpi.waitall %req1, %req2 : !comm.mpi.request, !comm.mpi.request
    return
}

// SHLO: func.func @main(%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiWaitall(%[[REQ1]], %[[REQ2]]) {has_side_effect = true} : (tensor<i64>, tensor<i64>) -> ()

// JIT-LABEL: llvm.mlir.global external constant @MPI_STATUSES_IGNORE
// JIT-SAME:                                                         () {addr_space = 0 : i32} : !llvm.ptr
// JIT-LABEL: llvm.func @MPI_Waitall
// JIT-SAME:                        (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_2
// JIT-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.mlir.constant(2 : i32) : i32
// JIT-NEXT:   %1 = llvm.alloca %0 x !llvm.ptr : (i32) -> !llvm.ptr 
// JIT-NEXT:   %2 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr 
// JIT-NEXT:   llvm.store %arg0, %2 : !llvm.ptr, !llvm.ptr 
// JIT-NEXT:   %3 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr 
// JIT-NEXT:   llvm.store %arg1, %3 : !llvm.ptr, !llvm.ptr 
// JIT-NEXT:   %4 = llvm.getelementptr %1[] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr 
// JIT-NEXT:   %5 = llvm.mlir.addressof @MPI_STATUSES_IGNORE : !llvm.ptr
// JIT-NEXT:   %6 = llvm.call @MPI_Waitall(%0, %4, %5) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>) {
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_2 (%[[REQ1]], %[[REQ2]]) : (tensor<i64>, tensor<i64>) -> ()
