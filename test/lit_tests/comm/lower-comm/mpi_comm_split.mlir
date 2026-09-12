// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt --lower-comm-to-jit %s | FileCheck %s --check-prefix=JIT

func.func @main(%comm : !comm.mpi.comm, %color : tensor<i32>, %key : tensor<i32>) -> !comm.mpi.comm {
    %0 = comm.mpi.comm_split %comm, %color, %key : (!comm.mpi.comm, tensor<i32>, tensor<i32>) -> !comm.mpi.comm
    return %0 : !comm.mpi.comm
}

// SHLO: func.func @main(%[[COMM:.*]]: tensor<i64>, %[[COLOR:.*]]: tensor<i32>, %[[KEY:.*]]: tensor<i32>) -> tensor<i64> {
// SHLO-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiCommSplit(%[[COMM]], %[[COLOR]], %[[KEY]]) : (tensor<i64>, tensor<i32>, tensor<i32>) -> tensor<i64>

// JIT-LABEL: llvm.func @MPI_Comm_split
// JIT-SAME:                           (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
// JIT-LABEL: llvm.func @enzymexla_jitwrap_MPI_Comm_split
// JIT-SAME:                                             (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// JIT-NEXT:   %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
// JIT-NEXT:   %1 = llvm.load %arg1 : !llvm.ptr -> i32
// JIT-NEXT:   %2 = llvm.load %arg2 : !llvm.ptr -> i32
// JIT-NEXT:   %3 = llvm.call @MPI_Comm_split(%0, %1, %2, %arg3) : (!llvm.ptr, i32, i32, !llvm.ptr) -> i32
// JIT-NEXT:   llvm.return
// JIT-NEXT: }
// JIT-LABEL: func.func @main
// JIT-SAME:                 (%arg0: tensor<i64>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i64> {
// JIT-NEXT:   %c = stablehlo.constant dense<-1> : tensor<i64>
// JIT-NEXT:   %0 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Comm_split (%arg0, %arg1, %arg2, %c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>]} : (tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>) -> tensor<i64>
// JIT-NEXT:   return %0 : tensor<i64>
// JIT-NEXT: }
