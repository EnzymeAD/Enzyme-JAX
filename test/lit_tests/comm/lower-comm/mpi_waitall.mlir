// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUSES_IGNORE=0x0" --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT
// RUN: enzymexlamlir-opt --map-symbol="symbols=MPI_STATUSES_IGNORE=0x0" --lower-comm-to-jit="mpi_request_width=32" %s | FileCheck %s --check-prefix=JIT32

func.func @main(%req1 : !comm.mpi.request, %req2 : !comm.mpi.request) {
    comm.mpi.waitall %req1, %req2 : !comm.mpi.request, !comm.mpi.request
    return
}

// SHLO: func.func @main(%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>) {
// SHLO-NEXT: stablehlo.custom_call @MpiWaitall(%[[REQ1]], %[[REQ2]]) {has_side_effect = true} : (tensor<i64>, tensor<i64>) -> ()

// JIT-LABEL: llvm.func @MPI_Waitall
// JIT-SAME:                        (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_2
// JIT-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.mlir.constant(2 : i32) : i32
// JIT-NEXT:   %1 = llvm.alloca %0 x i64 : (i32) -> !llvm.ptr 
// JIT-NEXT:   %2 = llvm.load %arg0 : !llvm.ptr -> i64
// JIT-NEXT:   %3 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, i64 
// JIT-NEXT:   llvm.store %2, %3 : i64, !llvm.ptr 
// JIT-NEXT:   %4 = llvm.load %arg1 : !llvm.ptr -> i64
// JIT-NEXT:   %5 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i64 
// JIT-NEXT:   llvm.store %4, %5 : i64, !llvm.ptr 
// JIT-NEXT:   %6 = llvm.mlir.constant(0 : i64) : i64
// JIT-NEXT:   %7 = llvm.inttoptr %6 : i64 to !llvm.ptr
// JIT-NEXT:   %8 = llvm.call @MPI_Waitall(%0, %1, %7) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%[[REQ1:.*]]: tensor<i64>, %[[REQ2:.*]]: tensor<i64>) {
// JIT-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_2 (%[[REQ1]], %[[REQ2]]) : (tensor<i64>, tensor<i64>) -> ()

// JIT32-LABEL: llvm.func @enzymexla_jitwrap_MPI_Waitall_2
// JIT32-SAME:                                            (%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
// JIT32-NEXT:   %0 = llvm.mlir.constant(2 : i32) : i32
// JIT32-NEXT:   %1 = llvm.alloca %0 x i32 : (i32) -> !llvm.ptr
// JIT32-NEXT:   %2 = llvm.load %arg0 : !llvm.ptr -> i32
// JIT32-NEXT:   %3 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, i32
// JIT32-NEXT:   llvm.store %2, %3 : i32, !llvm.ptr
// JIT32-NEXT:   %4 = llvm.load %arg1 : !llvm.ptr -> i32
// JIT32-NEXT:   %5 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, i32
// JIT32-NEXT:   llvm.store %4, %5 : i32, !llvm.ptr
// JIT32-LABEL: func.func @main
// JIT32-SAME:                 (%[[REQ1:.*]]: tensor<i32>, %[[REQ2:.*]]: tensor<i32>) {
// JIT32-NEXT: enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_2 (%[[REQ1]], %[[REQ2]]) : (tensor<i32>, tensor<i32>) -> ()
